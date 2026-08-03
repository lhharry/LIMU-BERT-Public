'''
Score EVERY checkpoint family under saved/bench_run16/ on the *unseen* jetson
leg NPY, window by window, and print one sorted table per family.

bench_run16 holds four method families, all trained on the merged dataset
(merged_10_20_merged_9cls, 9 classes) but with different architectures:

  1. bench_gru_merged_*/bench_R-GRU__*
        supervised ClassifierGRU, raw 6-dim window -> 9-class logits (no BERT).
  2. bert_classifier_base_gru_merged_*/..._finetune__*
        BERTClassifier (transformer + GRU head), transformer fine-tuned.
  3. bert_classifier_base_gru_merged_*/..._finetune-high-lr__*
        BERTClassifier, transformer fine-tuned with high learning rate.
        (finetune-high-lr vs finetune are byte-identical architectures; the finetune-high-lr flag
         only mattered during training, so at inference both load the same way.)
  4. classifier_base_gru_merged_*/..._separated__*
        standalone GRU head trained on cached FOUNDATION-BERT embeddings.
        Inference = foundation transformer (eval/no_grad) -> (N,seq,hidden)
        -> this head. The head's gru0 takes hidden=72, NOT raw 6-dim, so it
        MUST be fed BERT embeddings, never the raw window.

CLASS CORRESPONDENCE (the important bit, same as jetson_npy_test.py):
  merged 9-class : stand walk turn jog rampascent rampdescent
                   stairascent stairdescent sit-stand-transition   (ids 0..8)
  jetson 7-class : stand walk turn jog rampascent stairascent stairdescent
                   (ids 0..6)
ids do NOT line up (jetson stairascent=5/stairdescent=6 are 6/7 in the model),
so jetson ground truth is remapped to the model space *by NAME*, never by raw
id. rampdescent / sit-stand-transition get zero support but the model may still
(wrongly) predict them, lowering accuracy honestly.

All families consume the same per-window Preprocess4Normalization(6) of the raw
camargo-axis (*_xyz) jetson windows, so we normalize once and reuse it. DeepSense
is the one exception: benchmark.py trains it on FFTDataset, so it must be fed the
same fft(n=10) magnitude+phase features here (see fft_features / IMPORTANT below).
Edit the CONFIG dict to point at a different jetson version / foundation ckpt.

LEAKAGE GUARD: when npy_version is the SAME version the checkpoints were trained
on (detected from the subdir names), most of these windows were in each model's
training set, so scoring the full npy would inflate the memorizing baselines.
In that case each checkpoint is scored ONLY on its own seed's held-out test
split, reconstructed exactly like the benchmark's partition (see
bench_test_indices). When npy_version differs (true cross-version eval on
unseen data), the full npy is scored as before.
'''

import re
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import fetch_classifier, BERTClassifier, LIMUBertModel4Pretrain
from utils import Preprocess4Normalization, set_seeds

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CONFIG = {
    "run_dir": Path("saved/history/bench_53"),
    "gru_subdir": "bench_gru_jetson_leg_10_20_both_0103_xyz_both",
    "bert_subdir": "bert_classifier_base_gru_jetson_leg_10_20_both_0103_xyz_both",
    "sep_subdir": "classifier_base_gru_jetson_leg_10_20_both_0103_xyz_both",

    # class space + seq_len / sr / dimindo
    "dataset": "jetson_leg",
    "dataset_version": "10_20_both_0103_xyz_both",
    "bert_version": "v3",                 # base_v3 in config/limu_bert.json
    "classifier_version": "v3",           # gru_v3 in config/classifier.json
    "dcnn_version": "v1",                 # dcnn_v1 in config/classifier.json
    "deepsense_version": "v1",            # deepsense_v1 in config/classifier.json

    # foundation BERT that produced the separated-head training embeddings
    "foundation_ckpt": Path("saved/pretrain_base_merged_10_20_9cls/limu_bert_x_9cls_dapt_5e-4_3200_seed3431.pt"),

    # unseen jetson leg NPY (camargo axis order -> *_xyz variant matches training)
    "npy_dir": Path("dataset/jetson_leg"),
    "npy_version": "10_20_both_02_xyz_leg",

    # split geometry used by the benchmark run that produced the checkpoints;
    # needed to reconstruct each seed's held-out test split (leakage guard)
    "training_rate": 0.8,
    "vali_rate": 0.1,

    "batch_size": 128,
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_jetson_npy(npy_dir: Path, version: str):
    data = np.load(npy_dir / f"data_{version}.npy").astype(np.float32)
    label = np.load(npy_dir / f"label_{version}.npy")
    with open(npy_dir / "label_map.json") as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    gt_jetson = label[:, 0, 0].astype(int)
    return data, gt_jetson, id_to_name, label


def bench_test_indices(label_raw, seed, training_rate, vali_rate):
    """Reconstruct the benchmark's held-out test split on the training npy.

    Mirrors utils.prepare_classifier_dataset -> partition_and_reshape exactly:
    set_seeds(seed) is immediately followed by ONE np.random.shuffle over the
    window index, and the last (1 - training_rate - vali_rate) fraction is the
    test split. Windows whose per-frame activity labels are not uniform are
    dropped, matching merge_dataset(mode='all'). Returns original npy indices.
    """
    n = label_raw.shape[0]
    set_seeds(seed)
    arr = np.arange(n)
    np.random.shuffle(arr)
    train_num = int(n * training_rate)
    vali_num = int(n * vali_rate)
    test_idx = arr[train_num + vali_num:]
    act = label_raw[test_idx, :, 0]
    uniform = np.all(act == act[:, :1], axis=1)
    return test_idx[uniform]


def remap_gt_by_name(gt_jetson, id_to_name, label_names):
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


def fft_features(norm):
    """IMPORTANT: DeepSense trains on FFTDataset, not IMUDataset (benchmark.py
    branches on method == 'deepsense'), so it never sees a raw window. Mirror
    utils.FFTDataset.preprocess exactly -- fft over the time axis with n=10, then
    concat magnitude and phase along that axis -- or the model scores at chance.
    """
    f = np.fft.fft(norm, axis=1, n=10)
    return np.concatenate([np.abs(f), np.angle(f)], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# Forward helpers (one per family). All return per-window argmax predictions.
# -----------------------------------------------------------------------------
def _argmax_loader(tensor, model, batch_size, device):
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device), False)
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


def predict_supervised(norm, ck, classifier_cfg, label_num, bs, device, method="gru"):
    model = fetch_classifier(method, classifier_cfg, input=classifier_cfg.input, output=label_num)
    model.load_state_dict(torch.load(ck, map_location=device))
    model = model.to(device).eval()
    return _argmax_loader(torch.from_numpy(norm), model, bs, device)


def predict_bert(norm, ck, bert_cfg, classifier_cfg, label_num, bs, device):
    inner = fetch_classifier("gru", classifier_cfg, input=bert_cfg.hidden, output=label_num)
    model = BERTClassifier(bert_cfg, classifier=inner, frozen_bert=False)
    model.load_state_dict(torch.load(ck, map_location=device))
    model = model.to(device).eval()
    return _argmax_loader(torch.from_numpy(norm), model, bs, device)


def foundation_embeddings(norm, bert_cfg, foundation_ckpt, bs, device):
    """Run the finetune-high-lr foundation transformer -> (N, seq_len, hidden) embeddings."""
    bert = LIMUBertModel4Pretrain(bert_cfg, output_embed=True)
    bert.load_state_dict(torch.load(foundation_ckpt, map_location=device))
    bert = bert.to(device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(norm)), batch_size=bs, shuffle=False)
    embs = []
    with torch.no_grad():
        for (batch,) in loader:
            embs.append(bert(batch.to(device)).cpu())
    return torch.cat(embs, dim=0)          # (N, seq_len, hidden) on CPU


def predict_separated(embeddings, ck, classifier_cfg, bert_cfg, label_num, bs, device):
    head = fetch_classifier("gru", classifier_cfg, input=bert_cfg.hidden, output=label_num)
    head.load_state_dict(torch.load(ck, map_location=device))
    head = head.to(device).eval()
    return _argmax_loader(embeddings, head, bs, device)


# -----------------------------------------------------------------------------
# Reporting

# -----------------------------------------------------------------------------
def score(preds, gt_model, label_num):
    acc = float(np.mean(preds == gt_model))
    f1 = f1_score(gt_model, preds, labels=range(label_num), average="macro", zero_division=0)
    return acc, f1


def print_family(title, rows):
    if not rows:
        print(f"\n### {title}: no checkpoints found\n")
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    name_w = max(len(r[0]) for r in rows)
    print(f"\n### {title}  ({len(rows)} checkpoints)")
    print(f"{'checkpoint':{name_w}s} {'acc':>7s} {'macroF1':>8s}")
    print("-" * (name_w + 17))
    for name, acc, f1 in rows:
        print(f"{name:{name_w}s} {acc:7.3f} {f1:8.3f}")
    accs = np.array([r[1] for r in rows]); f1s = np.array([r[2] for r in rows])
    print("-" * (name_w + 17))
    print(f"{'mean':{name_w}s} {accs.mean():7.3f} {f1s.mean():8.3f}")
    print(f"{'best':{name_w}s} {accs.max():7.3f}  ({rows[0][0]})")
    return ("best", rows[0][0], accs.max(), f1s[0], accs.mean())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = CONFIG
    dataset_cfg = load_dataset_stats(cfg["dataset"], cfg["dataset_version"])
    if dataset_cfg is None:
        raise ValueError(f"Unknown dataset config: {cfg['dataset']}_{cfg['dataset_version']}")
    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    classifier_cfg = load_model_config("bench_gru", "gru", cfg["classifier_version"])
    bert_cfg = load_model_config("pretrain_base", "base", cfg["bert_version"])
    if classifier_cfg is None or bert_cfg is None:
        raise ValueError("Unable to load classifier/bert model config")
    dcnn_cfg = load_model_config("bench_dcnn", "dcnn", cfg["dcnn_version"])
    deepsense_cfg = load_model_config("bench_deepsense", "deepsense", cfg["deepsense_version"])
    seq_len, feature_count = dataset_cfg.seq_len, dataset_cfg.dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, gt_jetson, id_to_name, label_raw = load_jetson_npy(Path(cfg["npy_dir"]), cfg["npy_version"])
    if data.shape[1:] != (seq_len, feature_count):
        raise ValueError(f"jetson window shape {data.shape[1:]} != model "
                         f"(seq_len={seq_len}, dim={feature_count})")
    gt_model, keep = remap_gt_by_name(gt_jetson, id_to_name, label_names)
    dropped = int((~keep).sum())
    orig_idx = np.nonzero(keep)[0]
    data, gt_model = data[keep], gt_model[keep]
    norm = normalize_windows(data, feature_count)

    # Leakage guard: if the npy is the very version the checkpoints trained on,
    # ~80% of these windows sat in each model's training set. Restrict scoring
    # to the per-seed held-out test split in that case.
    # The npy DIRECTORY is part of the npy's identity, not just the version
    # string: dataset/jetson_leg_test/ holds test-only rebuilds that can carry
    # the very same version string as a training npy in dataset/jetson_leg/
    # (e.g. 10_20_both_01_xyz_pocket). Keying the guard on the directory name
    # keeps the old dataset/jetson_leg/ behaviour identical while never
    # mistaking a test-only npy for the checkpoints' training data.
    trained_suffix = f"{Path(cfg['npy_dir']).name}_{cfg['npy_version']}"
    mask_mode = cfg["gru_subdir"].endswith(trained_suffix)
    test_idx_cache = {}

    def eval_selector(ck_name):
        """Bool selector (over kept windows) = this ckpt's held-out test split; None = score all."""
        if not mask_mode:
            return None
        m = re.search(r"__seed(\d+)", ck_name)
        if m is None:
            raise ValueError(f"cannot parse seed from checkpoint name: {ck_name}")
        seed = int(m.group(1))
        if seed not in test_idx_cache:
            test_idx_cache[seed] = bench_test_indices(
                label_raw, seed, cfg["training_rate"], cfg["vali_rate"])
            n_test = int(np.isin(orig_idx, test_idx_cache[seed]).sum())
            print(f"[mask] seed {seed}: scoring {n_test} held-out test / {orig_idx.size} kept windows")
        return np.isin(orig_idx, test_idx_cache[seed])

    run = Path(cfg["run_dir"])
    gru_dir = run / cfg["gru_subdir"]
    bert_dir = run / cfg["bert_subdir"]
    sep_dir = run / cfg["sep_subdir"]

    print(f"Run dir   : {run}")
    print(f"jetson    : {cfg['npy_dir']}/data_{cfg['npy_version']}.npy  "
          f"({gt_model.size} scorable / {gt_jetson.size} total, dropped {dropped})")
    print(f"Model cls : {label_names}")
    present = {id_to_name[int(j)] for j in gt_jetson}
    jetson_present = [n for n in label_names if n in present]
    print(f"jetson cls: {jetson_present}")
    print(f"Foundation: {cfg['foundation_ckpt']}")
    print(f"Config    : seq_len {seq_len} | dim {feature_count} | hidden {bert_cfg.hidden} | device {device}")
    if mask_mode:
        print("[mask] npy_version matches the checkpoints' training version -> "
              "per-seed held-out test scoring (leakage guard ON)")
    else:
        print("[mask] npy_version differs from training version -> full-npy scoring (unseen data)")

    bs = cfg["batch_size"]
    summary = []

    def masked(arr, sel):
        return arr if sel is None else arr[sel]

    # 1. supervised R-GRU
    rows = []
    for ck in sorted(gru_dir.glob("R-GRU__*.pt")):
        sel = eval_selector(ck.name)
        preds = predict_supervised(masked(norm, sel), ck, classifier_cfg, label_num, bs, device)
        rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
    s = print_family("R-GRU (supervised, no BERT)", rows)
    if s: summary.append(("R-GRU", s))

    # 1b. supervised DCNN
    if dcnn_cfg is not None:
        dcnn_dir = run / cfg["gru_subdir"].replace("bench_gru_", "bench_dcnn_", 1)
        rows = []
        for ck in sorted(dcnn_dir.glob("DCNN__*.pt")):
            sel = eval_selector(ck.name)
            preds = predict_supervised(masked(norm, sel), ck, dcnn_cfg, label_num, bs, device, method="dcnn")
            rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
        s = print_family("DCNN (supervised, no BERT)", rows)
        if s: summary.append(("DCNN", s))

    # 1c. supervised DeepSense (fft features, NOT the raw window -- see fft_features)
    if deepsense_cfg is not None:
        deepsense_dir = run / cfg["gru_subdir"].replace("bench_gru_", "bench_deepsense_", 1)
        norm_fft = fft_features(norm)
        rows = []
        for ck in sorted(deepsense_dir.glob("DeepSense__*.pt")):
            sel = eval_selector(ck.name)
            preds = predict_supervised(masked(norm_fft, sel), ck, deepsense_cfg, label_num, bs, device, method="deepsense")
            rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
        s = print_family("DeepSense (supervised, no BERT)", rows)
        if s: summary.append(("DeepSense", s))

    # 2. finetune  (BERTClassifier)
    rows = []
    for ck in sorted(bert_dir.glob("*_finetune__*.pt")):
        sel = eval_selector(ck.name)
        preds = predict_bert(masked(norm, sel), ck, bert_cfg, classifier_cfg, label_num, bs, device)
        rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
    s = print_family("LIMU-BERT-X + GRU (finetune)", rows)
    if s: summary.append(("finetune", s))

    # 3. frozen (BERTClassifier, same load path)
    rows = []
    for ck in sorted(bert_dir.glob("*_frozen__*.pt")):
        sel = eval_selector(ck.name)
        preds = predict_bert(masked(norm, sel), ck, bert_cfg, classifier_cfg, label_num, bs, device)
        rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
    s = print_family("LIMU-BERT-X + GRU (frozen)", rows)
    if s: summary.append(("frozen", s))

    # 4. finetune-high-lr  (BERTClassifier, same load path)
    rows = []
    for ck in sorted(bert_dir.glob("*_finetune-high-lr__*.pt")):
        sel = eval_selector(ck.name)
        preds = predict_bert(masked(norm, sel), ck, bert_cfg, classifier_cfg, label_num, bs, device)
        rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
    s = print_family("LIMU-BERT-X + GRU (finetune-high-lr)", rows)
    if s: summary.append(("finetune-high-lr", s))

    # 5. separated  (foundation embeddings -> standalone head)
    rows = []
    sep_ckpts = sorted(sep_dir.glob("*_separated__*.pt"))
    if sep_ckpts:
        embeddings = foundation_embeddings(norm, bert_cfg, Path(cfg["foundation_ckpt"]), bs, device)
        for ck in sep_ckpts:
            sel = eval_selector(ck.name)
            emb = embeddings if sel is None else embeddings[torch.as_tensor(np.nonzero(sel)[0], dtype=torch.long)]
            preds = predict_separated(emb, ck, classifier_cfg, bert_cfg, label_num, bs, device)
            rows.append((ck.name, *score(preds, masked(gt_model, sel), label_num)))
    s = print_family("LIMU-BERT-X + GRU (separated)", rows)
    if s: summary.append(("separated", s))

    # cross-family summary
    print("\n" + "=" * 52)
    print("SUMMARY (best checkpoint per family)")
    print(f"{'family':12s} {'best_acc':>9s} {'mean_acc':>9s}   best_ckpt")
    print("-" * 52)
    for fam, (_, name, best_acc, best_f1, mean_acc) in summary:  # best_f1 = f1 of the best-acc ckpt
        print(f"{fam:12s} {best_acc:9.3f} {mean_acc:9.3f}   {name},{best_acc:9.3f} {best_f1:9.3f}  ")


if __name__ == "__main__":
    main()
