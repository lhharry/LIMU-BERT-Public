'''
Run the fine-tuned LIMU-BERT+GRU classifier on the jetson real recordings and
score the predictions against the activity implied by each trial's folder name.

How it works
------------
Each jetson trial lives in DATA/jetson/<HH_MM_SS>_<class>_<position>/ (and the
stair trials under DATA/jetson/stair/...), holding accelerometers.csv and
gyroscopes.csv with columns Time, Left_{x,y,z}, Right_{x,y,z}. There is no
per-sample label, so the *folder class* is used as the ground truth for the whole
trial. Because every recording starts and ends with a few seconds of standing,
those edge windows are trimmed before scoring (TRIM_SECONDS) so the "stand"
padding does not penalise the active-class accuracy.

Pipeline per trial:  pick a thigh side (Left/Right) -> [accel xyz, gyro xyz] in
native units (accel m/s^2, gyro rad/s, exactly what the model was trained on) ->
mean-pool downsample to 10 Hz -> trim edges -> 2 s windows (seq_len=20) ->
Preprocess4Normalization -> classifier -> compare argmax to the folder class.

Config: edit the small CONFIG dict below. `folder=None` runs every trial under
`jetson_root`; set it to one folder name to test a single recording.
'''

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import BERTClassifier, fetch_classifier
from utils import Preprocess4Normalization
from inference.downsampling import down_sample

# -----------------------------------------------------------------------------
# Config  (only the knobs you actually change; everything else is derived from
#          the dataset/model config so it cannot drift out of sync)
# -----------------------------------------------------------------------------
CONFIG = {
    # what to run
    "jetson_root": Path(r"D:\01_Code\DATA\jetson"),
    "folder": None,                 # None = all trials under jetson_root; or a
                                    # folder name e.g. "11_34_28_jog_leg"
    "sides": ["left", "right"],     # thigh sensor(s) to feed the single-IMU model
                                    # ("left" / "right" / "average")
    "trim_seconds": 3.0,            # drop this many s of edge "stand" before scoring

    # model checkpoint (this fixes seq_len, sr, #features, #classes via its config)
    "model_path": Path("saved/bert_classifier_base_gru_camargo_10_20_dense_8cls/limu_gru_dapt.pt"),
    "dataset": "camargo",
    "dataset_version": "10_20_dense_8cls",
    "bert_version": "v3",
    "classifier_version": "v3",

    "stride": None,                 # None = non-overlapping windows (= seq_len)
    "batch_size": 128,
}

# jetson folder class token -> dense activity name (must match the model's classes)
JETSON_TOKEN_TO_DENSE = {
    "walk": "walk", "jog": "jog",
    "rampup": "rampascent", "rampdown": "rampdescent",
    "stairup": "stairascent", "stairdown": "stairdescent",
    "turnleft": "turn", "turnright": "turn",
}

SIDE_COLS = {
    "left": ["Left_x", "Left_y", "Left_z"],
    "right": ["Right_x", "Right_y", "Right_z"],
}


# -----------------------------------------------------------------------------
# Jetson trial loading
# -----------------------------------------------------------------------------
def resolve_folders(cfg):
    root = Path(cfg["jetson_root"])
    if cfg["folder"]:
        p = Path(cfg["folder"])
        if not p.is_absolute():
            p = root / cfg["folder"]
        return [p]
    folders = []
    for dirpath, _dirs, files in os.walk(root):
        if "accelerometers.csv" in files and "gyroscopes.csv" in files:
            folders.append(Path(dirpath))
    return sorted(folders)


def parse_folder(folder: Path):
    """folder name '<HH>_<MM>_<SS>_<class>_<position>[_variant]' -> tokens."""
    toks = folder.name.split("_")
    class_token = toks[3].lower() if len(toks) > 3 else ""
    position = toks[4].lower() if len(toks) > 4 else ""
    variant = "_".join(toks[5:]) if len(toks) > 5 else ""
    return class_token, position, variant


def estimate_sr(folder: Path, fallback: float = 63.0) -> float:
    df = pd.read_csv(folder / "accelerometers.csv", usecols=["Time"])
    secs = []
    for x in df["Time"].astype(str):
        try:
            h, m, s = x.split(":")
            secs.append(int(h) * 3600 + int(m) * 60 + float(s))
        except Exception:
            continue
    if len(secs) < 2:
        return fallback
    dt = np.diff(np.asarray(secs))
    dt = dt[(dt > 0) & (dt < 1)]
    return float(1.0 / np.median(dt)) if dt.size else fallback


def load_jetson_side(folder: Path, side: str) -> np.ndarray:
    """Return (T, 6) array [ax,ay,az,gx,gy,gz] in native units for one thigh side."""
    acc = pd.read_csv(folder / "accelerometers.csv")
    gyr = pd.read_csv(folder / "gyroscopes.csv")
    n = min(len(acc), len(gyr))
    acc, gyr = acc.iloc[:n], gyr.iloc[:n]
    if side == "average":
        a = (acc[SIDE_COLS["left"]].to_numpy(float) + acc[SIDE_COLS["right"]].to_numpy(float)) * 0.5
        g = (gyr[SIDE_COLS["left"]].to_numpy(float) + gyr[SIDE_COLS["right"]].to_numpy(float)) * 0.5
    else:
        a = acc[SIDE_COLS[side]].to_numpy(float)
        g = gyr[SIDE_COLS[side]].to_numpy(float)
    return np.hstack([a, g]).astype(np.float32)   # accel m/s^2, gyro rad/s


# -----------------------------------------------------------------------------
# Windowing / model  (shared with the rest of the inference tooling)
# -----------------------------------------------------------------------------
def window_features(features, window_size, stride):
    if features.shape[0] < window_size:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32)
    windows = [features[s:s + window_size]
               for s in range(0, features.shape[0] - window_size + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


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
# Per-trial evaluation
# -----------------------------------------------------------------------------
def evaluate_trial(folder, side, model, cfg, seq_len, stride, target_sr,
                   feature_count, label_names):
    class_token, position, variant = parse_folder(folder)
    dense = JETSON_TOKEN_TO_DENSE.get(class_token)
    if dense is None or dense not in label_names:
        return {"folder": folder.name, "side": side, "skip": f"unknown class '{class_token}'"}
    gt = label_names.index(dense)

    raw_sr = estimate_sr(folder)
    feats = load_jetson_side(folder, side)              # (T_raw, 6) native units
    feats = down_sample(feats, raw_sr, target_sr).astype(np.float32)  # -> 10 Hz

    n = feats.shape[0]
    trim = int(round(cfg["trim_seconds"] * target_sr))
    windows = window_features(feats, seq_len, stride)
    if windows.shape[0] == 0:
        return {"folder": folder.name, "side": side,
                "skip": f"too short after downsample (n={n} < seq_len={seq_len})"}

    norm = normalize_sequence_data(windows, feature_count)
    device = next(model.parameters()).device
    preds = predict(norm, model, cfg["batch_size"], device)

    # a window is "active" (scored) if it lies fully inside the trimmed region
    starts = np.arange(0, n - seq_len + 1, stride)[:len(preds)]
    active = (starts >= trim) & (starts + seq_len <= n - trim)
    scored = preds[active] if active.any() else preds   # fall back to all if trim ate everything

    acc = float(np.mean(scored == gt)) if scored.size else float("nan")
    uniq, cnt = np.unique(preds, return_counts=True)
    dist = {label_names[c]: int(n_) for c, n_ in zip(uniq, cnt)}
    maj = int(uniq[np.argmax(cnt)]) if uniq.size else -1

    return {
        "folder": folder.name, "class": dense, "position": position, "side": side,
        "raw_sr": raw_sr, "n_down": n, "n_windows": int(preds.size),
        "n_scored": int(scored.size), "accuracy": acc,
        "gt": gt, "majority": label_names[maj] if maj >= 0 else "?",
        "scored_preds": scored, "dist": dist,
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_report(results, label_names):
    print("\n=== Per-trial results "
          "(accuracy = scored windows predicted as the folder class) ===")
    hdr = f"{'folder':30s} {'gt':12s} {'side':6s} {'sr':>5s} {'win':>4s} " \
          f"{'scd':>4s} {'acc':>6s}  majority  distribution"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "skip" in r:
            print(f"{r['folder']:30s}  [skip] {r['skip']}")
            continue
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(r["dist"].items(),
                                                       key=lambda kv: -kv[1]))
        print(f"{r['folder']:30s} {r['class']:12s} {r['side']:6s} "
              f"{r['raw_sr']:5.1f} {r['n_windows']:4d} {r['n_scored']:4d} "
              f"{r['accuracy']:6.3f}  {r['majority']:11s} {dist}")

    scored_results = [r for r in results if "skip" not in r and r["n_scored"] > 0]
    if not scored_results:
        return

    # per-class accuracy over pooled scored windows
    print("\n=== Per-class accuracy (scored windows pooled over sides/trials) ===")
    by_class = {}
    for r in scored_results:
        p, g = r["scored_preds"], r["gt"]
        c, t = by_class.setdefault(r["class"], [0, 0])
        by_class[r["class"]] = [c + int(np.sum(p == g)), t + p.size]
    print(f"{'class':14s} {'windows':>8s} {'accuracy':>9s}")
    print("-" * 33)
    accs = []
    for cls in label_names:
        if cls in by_class:
            corr, tot = by_class[cls]
            a = corr / tot if tot else float("nan")
            accs.append(a)
            print(f"{cls:14s} {tot:8d} {a:9.3f}")

    all_correct = sum(c for c, _ in by_class.values())
    all_total = sum(t for _, t in by_class.values())
    print(f"\nOverall window accuracy : {all_correct / all_total:.3f} "
          f"({all_correct}/{all_total})")
    print(f"Macro (per-class) accuracy: {np.nanmean(accs):.3f}")


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
    target_sr = dataset_cfg.sr
    feature_count = dataset_cfg.dimension
    stride = cfg["stride"] or seq_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(device, label_num, cfg["bert_version"],
                           cfg["classifier_version"], Path(cfg["model_path"]))

    folders = resolve_folders(cfg)
    print(f"Model     : {cfg['model_path']}")
    print(f"Classes   : {label_names}")
    print(f"Target SR : {target_sr} Hz | seq_len {seq_len} | stride {stride} | "
          f"trim {cfg['trim_seconds']}s | device {device}")
    print(f"Trials    : {len(folders)} folder(s), sides={cfg['sides']}")

    results = []
    for folder in folders:
        _c, _p, variant = parse_folder(folder)
        if "zeroed" in variant:
            print(f"[skip] zeroed variant: {folder.name}")
            continue
        for side in cfg["sides"]:
            try:
                results.append(evaluate_trial(
                    folder, side, model, cfg, seq_len, stride, target_sr,
                    feature_count, label_names))
            except Exception as exc:
                results.append({"folder": folder.name, "side": side, "skip": repr(exc)})

    print_report(results, label_names)


if __name__ == "__main__":
    main()
