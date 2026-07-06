'''
Score the fine-tuned LIMU-BERT+GRU classifier on the *preprocessed* jetson NPY,
window by window.

How it works
------------
dataset/jetson_leg.py turns the raw jetson leg recordings into the LIMU-BERT NPY
format under dataset/jetson_leg/:
    data_<version>.npy   (N, seq_len, 6)  float32  -- already 10 Hz, windowed,
                                                       axis-remapped to camargo
                                                       order, NOT normalized.
    label_<version>.npy  (N, seq_len, 2)  float    -- [activity_id, user_id];
                                                       activity id is constant
                                                       within a window.
    label_map.json       {name: id}                -- jetson 7-class space.

Unlike the old CSV path, every window already carries its own per-segment label
(stand -> activity -> stand, from label.csv), so there is no folder-name guessing
and no edge trimming. We just normalize each window, run the model, and compare
per window.

Label-space note: the jetson NPY uses a 7-class space (no "rampdescent"), while
the camargo model outputs 8 classes. Ids 5/6 differ between the two spaces, so
ground truth is remapped to the model space *by name*, never by raw id.

Pipeline:  load data/label NPY -> remap gt to model classes by name ->
Preprocess4Normalization per window -> classifier -> accuracy / macro-F1 /
confusion matrix over the model's class space.

Config: edit the small CONFIG dict below.
'''

import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import BERTClassifier, fetch_classifier
from utils import Preprocess4Normalization

# -----------------------------------------------------------------------------
# Config  (only the knobs you actually change; everything else is derived from
#          the dataset/model config so it cannot drift out of sync)
# -----------------------------------------------------------------------------
CONFIG = {
    # preprocessed jetson NPY (produced by dataset/jetson_leg.py)
    "npy_dir": Path("dataset/jetson_leg"),  # directory containing data_/label_<version>.npy
    "npy_version": "10_20_both_02_xyz_leg_AB02",        # data_/label_<version>.npy in npy_dir

    # model checkpoint (this fixes seq_len, sr, #features, #classes via its config)
    "model_path": Path("saved/best/BERTGRU_align_1e-3_Pocket_finetune-high-lr__lr0.2__seed3431_0.804.pt"),
    "dataset": "jetson_leg",            # defines the model's class space + seq_len/sr/dim
    "dataset_version": "10_20_both_02_xyz_leg_AB02",
    "bert_version": "v3",
    "classifier_version": "v3",

    "batch_size": 128,

    # optional: restrict evaluation to specific subjects, e.g. ["AB02"].
    # None / [] -> all subjects in the NPY.
    "subjects": None,
}


# -----------------------------------------------------------------------------
# Jetson NPY loading
# -----------------------------------------------------------------------------
def load_jetson_npy(npy_dir: Path, version: str, subjects=None, user_label=None):
    """Return (data (N,seq,6) float32, gt_jetson (N,) int, id_to_name dict).

    subjects: optional list of AB names (e.g. ["AB02"]) to keep. Windows whose
    user_id (label[:, 0, 1]) maps to a non-listed subject are dropped. The
    user_id -> AB mapping comes from `user_label` (the version's user_label list
    in dataset/data_config.json), indexed by user_id.
    """
    data = np.load(npy_dir / f"data_{version}.npy").astype(np.float32)   # (N,seq,6)
    label = np.load(npy_dir / f"label_{version}.npy")                    # (N,seq,2)
    with open(npy_dir / "label_map.json") as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}

    if subjects:
        if not user_label:
            raise ValueError("subjects filter needs user_label (the version's "
                             "user_label list from data_config.json)")
        want = {s.upper() for s in subjects}
        keep_ids = {i for i, name in enumerate(user_label) if name.upper() in want}
        missing = want - {user_label[i].upper() for i in keep_ids}
        if missing:
            raise ValueError(f"subjects {sorted(missing)} not in user_label {user_label}")
        user_ids = label[:, 0, 1].astype(int)
        mask = np.isin(user_ids, list(keep_ids))
        data, label = data[mask], label[mask]

    gt_jetson = label[:, 0, 0].astype(int)        # per-window activity id
    return data, gt_jetson, id_to_name


# -----------------------------------------------------------------------------
# Model  (shared with the rest of the inference tooling)
# -----------------------------------------------------------------------------
def normalize_sequence_data(data, feature_count):
    normalizer = Preprocess4Normalization(feature_count)
    return np.stack([normalizer(sample) for sample in data], axis=0).astype(np.float32)


def build_model(device, label_num, bert_version, classifier_version, model_path: Path):
    bert_cfg = load_model_config("pretrain_base", "base", bert_version)
    classifier_cfg = load_model_config("classifier_base_gru", "gru", classifier_version)
    if bert_cfg is None or classifier_cfg is None:
        raise ValueError("Unable to load bert/classifier model config")
    inner = fetch_classifier("gru", classifier_cfg, input=bert_cfg.hidden, output=label_num)
    model = BERTClassifier(bert_cfg, classifier=inner, frozen_bert=False).to(device)
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, bert_cfg


def predict(data, model, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device), False)
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.int64)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(data, gt_jetson, id_to_name, model, cfg, seq_len, feature_count, label_names):
    """Normalize -> predict -> return (gt_model (M,), preds (M,), dropped count)."""
    if data.shape[1:] != (seq_len, feature_count):
        raise ValueError(f"NPY window shape {data.shape[1:]} != expected "
                         f"(seq_len={seq_len}, feature_count={feature_count})")

    # remap jetson activity id -> model class index *by name*
    gt_model = np.full(gt_jetson.shape[0], -1, dtype=np.int64)
    for i, jid in enumerate(gt_jetson):
        name = id_to_name.get(int(jid))
        if name is not None and name in label_names:
            gt_model[i] = label_names.index(name)
    keep = gt_model >= 0
    dropped = int((~keep).sum())

    data, gt_model = data[keep], gt_model[keep]
    norm = normalize_sequence_data(data, feature_count)
    device = next(model.parameters()).device
    preds = predict(norm, model, cfg["batch_size"], device)
    return gt_model, preds, dropped


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_report(gt_model, preds, label_names, label_num):
    if gt_model.size == 0:
        print("\nNo scorable windows.")
        return

    acc = float(np.mean(preds == gt_model))
    macro_f1 = f1_score(gt_model, preds, labels=range(label_num),
                        average="macro", zero_division=0)

    # per-class accuracy / support
    print("\n=== Per-class accuracy (window-level) ===")
    print(f"{'class':14s} {'support':>8s} {'accuracy':>9s}")
    print("-" * 33)
    for c, cls in enumerate(label_names):
        mask = gt_model == c
        tot = int(mask.sum())
        if tot == 0:
            continue
        a = float(np.mean(preds[mask] == c))
        print(f"{cls:14s} {tot:8d} {a:9.3f}")

    print(f"\nOverall accuracy : {acc:.3f} "
          f"({int(np.sum(preds == gt_model))}/{gt_model.size})")
    print(f"Macro F1         : {macro_f1:.3f}")

    # confusion matrix (rows = true, cols = pred) over the model's class space
    cm = confusion_matrix(gt_model, preds, labels=range(label_num))
    short = [c[:6] for c in label_names]
    print("\n=== Confusion matrix (rows = true, cols = pred) ===")
    corner = "true\\pred"
    print(f"{corner:>12s} " + " ".join(f"{s:>6s}" for s in short))
    for i, cls in enumerate(label_names):
        row = " ".join(f"{v:6d}" for v in cm[i])
        print(f"{cls:>12s} {row}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = CONFIG
    dataset_cfg = load_dataset_stats(cfg["dataset"], cfg["dataset_version"])
    if dataset_cfg is None:
        raise ValueError(f"Unable to load dataset config: {cfg['dataset']}_{cfg['dataset_version']}")
    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    if not label_names or label_num <= 0:
        raise ValueError("Unable to resolve class names for label index 0.")

    seq_len = dataset_cfg.seq_len
    feature_count = dataset_cfg.dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(device, label_num, cfg["bert_version"],
                           cfg["classifier_version"], Path(cfg["model_path"]))

    data, gt_jetson, id_to_name = load_jetson_npy(
        Path(cfg["npy_dir"]), cfg["npy_version"],
        subjects=cfg.get("subjects"), user_label=dataset_cfg.user_label)

    subj_note = f"  (subjects {cfg['subjects']})" if cfg.get("subjects") else ""
    print(f"Model     : {cfg['model_path']}")
    print(f"NPY       : {cfg['npy_dir']}/data_{cfg['npy_version']}.npy  ({data.shape[0]} windows){subj_note}")
    print(f"Classes   : {label_names}")
    print(f"Config    : seq_len {seq_len} | dim {feature_count} | device {device}")

    gt_model, preds, dropped = evaluate(
        data, gt_jetson, id_to_name, model, cfg, seq_len, feature_count, label_names)
    if dropped:
        print(f"[warn] dropped {dropped} window(s) whose jetson class is absent "
              f"from the model's label space")

    print_report(gt_model, preds, label_names, label_num)


if __name__ == "__main__":
    main()
