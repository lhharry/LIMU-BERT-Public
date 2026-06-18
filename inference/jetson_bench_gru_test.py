'''
Score every supervised R-GRU benchmark checkpoint in
    saved/bench_gru_merged_10_20_9cls/
on the *unseen* jetson leg NPY, window by window, and print a sorted table.

Context
-------
The bench_R-GRU__lr*__seed*.pt files are plain ClassifierGRU state_dicts trained
by benchmark.py on the merged dataset (merged_10_20_merged_9cls). They take a
raw (B, seq_len, 6) IMU window (no BERT) and output 9-class logits:

    merged 9-class : stand walk turn jog rampascent rampdescent
                     stairascent stairdescent sit-stand-transition   (ids 0..8)

The jetson leg NPY (dataset/jetson_leg/, produced by jetson_leg.py) is a 7-class
space WITHOUT rampdescent / sit-stand-transition:

    jetson 7-class : stand walk turn jog rampascent stairascent stairdescent
                     (ids 0..6)

CLASS CORRESPONDENCE (the important bit):
ids do NOT line up -- jetson stairascent=5/stairdescent=6 are 6/7 in the model.
So ground truth is remapped to the model's class space *by name*, never by raw
id (exactly like inference/jetson_npy_test.py). The two model classes jetson
never contains (rampdescent, sit-stand-transition) simply receive zero support;
the model may still emit them as (wrong) predictions, which lowers accuracy
honestly.

Both jetson NPY and the merged training data use the camargo axis order, so we
read the *_xyz jetson variant and only per-window normalize before predicting.

Pipeline: load jetson data/label NPY -> remap GT by name -> for each checkpoint:
Preprocess4Normalization per window -> ClassifierGRU -> acc / macro-F1.
Edit the CONFIG dict below to point at a different folder / jetson version.
'''

import sys
import json
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import fetch_classifier
from utils import Preprocess4Normalization

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CONFIG = {
    # folder of supervised ClassifierGRU checkpoints to score
    "model_dir": Path("saved/bench_gru_merged_10_20_9cls"),
    "model_glob": "*.pt",

    # the model's class space + seq_len / sr / dim come from this dataset config
    "dataset": "merged",
    "dataset_version": "10_20_merged_9cls",
    "classifier_version": "v3",            # gru_v3 in config/classifier.json (R-GRU)

    # unseen jetson leg NPY (camargo axis order -> *_xyz variant)
    "npy_dir": Path("dataset/jetson_leg"),
    "npy_version": "10_20_both_yxz",

    "batch_size": 128,
    "show_confusion_for_best": True,       # print confusion matrix of the top model
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_jetson_npy(npy_dir: Path, version: str):
    """Return (data (N,seq,6) float32, gt_jetson (N,) int, id_to_name dict)."""
    data = np.load(npy_dir / f"data_{version}.npy").astype(np.float32)
    label = np.load(npy_dir / f"label_{version}.npy")
    with open(npy_dir / "label_map.json") as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    gt_jetson = label[:, 0, 0].astype(int)
    return data, gt_jetson, id_to_name


def remap_gt_by_name(gt_jetson, id_to_name, label_names):
    """jetson activity id -> model class index, matched by class NAME."""
    gt_model = np.full(gt_jetson.shape[0], -1, dtype=np.int64)
    for i, jid in enumerate(gt_jetson):
        name = id_to_name.get(int(jid))
        if name is not None and name in label_names:
            gt_model[i] = label_names.index(name)
    keep = gt_model >= 0
    return gt_model, keep


def normalize_windows(data, feature_count):
    norm = Preprocess4Normalization(feature_count)
    return np.stack([norm(s) for s in data], axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_gru(model_cfg, label_num, device, ckpt_path: Path):
    model = fetch_classifier("gru", model_cfg, input=model_cfg.input, output=label_num)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def predict(data, model, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device), False)
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = CONFIG
    dataset_cfg = load_dataset_stats(cfg["dataset"], cfg["dataset_version"])
    if dataset_cfg is None:
        raise ValueError(f"Unknown dataset config: {cfg['dataset']}_{cfg['dataset_version']}")
    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    model_cfg = load_model_config("bench_gru", "gru", cfg["classifier_version"])
    if model_cfg is None:
        raise ValueError(f"Unknown classifier config gru_{cfg['classifier_version']}")
    seq_len, feature_count = dataset_cfg.seq_len, dataset_cfg.dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- jetson data, prepared once and shared across all checkpoints ---
    data, gt_jetson, id_to_name = load_jetson_npy(Path(cfg["npy_dir"]), cfg["npy_version"])
    if data.shape[1:] != (seq_len, feature_count):
        raise ValueError(f"jetson window shape {data.shape[1:]} != model "
                         f"(seq_len={seq_len}, dim={feature_count})")
    gt_model, keep = remap_gt_by_name(gt_jetson, id_to_name, label_names)
    dropped = int((~keep).sum())
    data, gt_model = data[keep], gt_model[keep]
    norm = normalize_windows(data, feature_count)

    ckpts = sorted(Path(cfg["model_dir"]).glob(cfg["model_glob"]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {cfg['model_dir']}/{cfg['model_glob']}")

    print(f"Model dir : {cfg['model_dir']}  ({len(ckpts)} checkpoints)")
    print(f"jetson    : {cfg['npy_dir']}/data_{cfg['npy_version']}.npy  "
          f"({gt_model.size} scorable / {gt_jetson.size} total windows, dropped {dropped})")
    print(f"Model cls : {label_names}")
    jetson_present = sorted({id_to_name[int(j)] for j in gt_jetson})
    print(f"jetson cls: {jetson_present}  (rampdescent / sit-stand-transition absent -> 0 support)")
    print(f"Config    : seq_len {seq_len} | dim {feature_count} | device {device}\n")

    rows = []
    for ck in ckpts:
        model = build_gru(model_cfg, label_num, device, ck)
        preds = predict(norm, model, cfg["batch_size"], device)
        acc = float(np.mean(preds == gt_model))
        f1 = f1_score(gt_model, preds, labels=range(label_num), average="macro", zero_division=0)
        rows.append((ck.name, acc, f1, preds))

    rows.sort(key=lambda r: r[1], reverse=True)

    name_w = max(len(r[0]) for r in rows)
    print(f"{'checkpoint':{name_w}s} {'acc':>7s} {'macroF1':>8s}")
    print("-" * (name_w + 17))
    for name, acc, f1, _ in rows:
        print(f"{name:{name_w}s} {acc:7.3f} {f1:8.3f}")

    accs = np.array([r[1] for r in rows])
    f1s = np.array([r[2] for r in rows])
    print("-" * (name_w + 17))
    print(f"{'mean':{name_w}s} {accs.mean():7.3f} {f1s.mean():8.3f}")
    print(f"{'std':{name_w}s} {accs.std():7.3f} {f1s.std():8.3f}")
    print(f"{'best':{name_w}s} {accs.max():7.3f}  ({rows[0][0]})")

    if cfg["show_confusion_for_best"]:
        best_name, best_acc, _, best_preds = rows[0]
        print(f"\n=== Per-class accuracy of best model ({best_name}) ===")
        print(f"{'class':22s} {'support':>8s} {'acc':>7s}")
        print("-" * 39)
        for c, cls in enumerate(label_names):
            mask = gt_model == c
            tot = int(mask.sum())
            if tot == 0:
                continue
            print(f"{cls:22s} {tot:8d} {float(np.mean(best_preds[mask] == c)):7.3f}")

        cm = confusion_matrix(gt_model, best_preds, labels=range(label_num))
        short = [c[:6] for c in label_names]
        print("\n=== Confusion matrix (rows = true, cols = pred) ===")
        print(f"{'true\\pred':>12s} " + " ".join(f"{s:>6s}" for s in short))
        for i, cls in enumerate(label_names):
            print(f"{cls:>12s} " + " ".join(f"{v:6d}" for v in cm[i]))


if __name__ == "__main__":
    main()
