"""
Molinaro hip-exo dataset -> LIMU-BERT NPY format

Walks the Molinaro directory tree:

    <input_dir>/
        AB10/
            LG_C0p0_S1p25_UC/exo.csv
            RA_C5p0_S1p0_UC/exo.csv
            ST_C0p0_S0p0_UC/exo.csv
            ...
        AB11/...

Activity label = the FIRST underscore-separated token in the trial folder name
(e.g. "LG_C0p0_S1p25_UC" -> "LG"). Reads only `exo.csv`, extracts the 6 LEFT-thigh
IMU columns, optionally rotates into your 'self' frame, downsamples 200 Hz -> 20 Hz,
slices into 120-sample non-overlapping windows, and saves data_20_120.npy +
label_20_120.npy in the format LIMU-BERT expects.

Output shapes:
    data_20_120.npy   (N, 120, 6)   float32
    label_20_120.npy  (N, 120, 1)   int32

Recommended phases for fine-tune (per the dataset README + paper):
    --participants AB10-AB14 AB25-AB34       # Phase 2 + Phase 4

Recommended modes (steady-state HAR, drop transitions):
    --include_modes LG RA RD SA SD ST

Usage:
    python preprocess_molinaro.py \\
        --input_dir   /path/to/molinaro_root/ \\
        --output_dir  ./LIMU-BERT-Public/dataset/molinaro/ \\
        --participants AB10 AB11 AB12 AB13 AB14 \\
                       AB25 AB26 AB27 AB28 AB29 AB30 AB31 AB32 AB33 AB34 \\
        --include_modes LG RA RD SA SD ST \\
        --rotation gravity_R.npy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


# === LEFT thigh IMU columns inside exo.csv ===
FEATURE_COLS = [
    'thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l',  # acc m/s^2
    'thigh_gyro_x_l',  'thigh_gyro_y_l',  'thigh_gyro_z_l',   # gyro rad/s
]


def discover_trials(root_dir, participants=None, include_modes=None):
    """Walk <root>/<AB##>/<Trial_Name>/exo.csv. Returns list of (path, mode, subject)."""
    out = []
    root = Path(root_dir)
    subj_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith('AB'))
    if participants:
        wanted = set(participants)
        subj_dirs = [d for d in subj_dirs if d.name in wanted]
    for subj in subj_dirs:
        for trial in sorted(t for t in subj.iterdir() if t.is_dir()):
            mode = trial.name.split('_')[0]   # "LG_C0p0_S1p25_UC" -> "LG"
            if include_modes and mode not in include_modes:
                continue
            exo = trial / 'exo.csv'
            if exo.exists():
                out.append((exo, mode, subj.name))
    return out


def load_exo_csv(path):
    """Read exo.csv, return X (T,6) float32 keeping only rows where ALL 6 IMU cols are finite."""
    df = pd.read_csv(path)   # exo.csv is comma-separated
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    X = df[FEATURE_COLS].values.astype(np.float32)
    finite = np.all(np.isfinite(X), axis=1)
    return X[finite]


def apply_rotation(X, R):
    """Rotate acc (cols 0:3) and gyro (cols 3:6) by the same R (shared body frame)."""
    X = X.copy()
    X[:, 0:3] = X[:, 0:3] @ R.T
    X[:, 3:6] = X[:, 3:6] @ R.T
    return X


def downsample(X, src_fs, tgt_fs):
    """Anti-aliased polyphase downsample of X. Returns float32."""
    if abs(src_fs - tgt_fs) < 1e-3:
        return X.astype(np.float32)
    factor = int(round(src_fs / tgt_fs))
    if abs(src_fs / tgt_fs - factor) > 1e-3:
        raise ValueError(f"src_fs/tgt_fs={src_fs}/{tgt_fs} is not an integer ratio")
    return resample_poly(X, up=1, down=factor, axis=0).astype(np.float32)


def make_windows(X, label_int, window=120, stride=120):
    """Non-overlapping windows. All timesteps in a window get the same label."""
    n = X.shape[0]
    if n < window:
        return (np.empty((0, window, X.shape[1]), dtype=np.float32),
                np.empty((0, window),                dtype=np.int32))
    n_win = (n - window) // stride + 1
    Xw = np.stack([X[i*stride:i*stride+window] for i in range(n_win)], axis=0).astype(np.float32)
    yw = np.full((n_win, window), label_int, dtype=np.int32)
    return Xw, yw


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_dir',     required=True, help='Root containing AB10/, AB11/, ...')
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--participants',  nargs='*', default=None,
                   help='Subset of participant codes to include (default: all AB*)')
    p.add_argument('--include_modes', nargs='*', default=None,
                   help='Subset of activity modes (default: all). e.g. LG RA RD SA SD ST')
    p.add_argument('--src_fs',        type=float, default=200.0)
    p.add_argument('--tgt_fs',        type=float, default=20.0)
    p.add_argument('--window_size',   type=int,   default=120)
    p.add_argument('--rotation',      default=None,
                   help='Path to 3x3 .npy rotation matrix from compute_alignment.py')
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    trials = discover_trials(args.input_dir, args.participants, args.include_modes)
    if not trials:
        raise SystemExit(f"No trials found under {args.input_dir} matching filters.")
    print(f"Found {len(trials)} trial(s) across "
          f"{len({s for _,_,s in trials})} participant(s).")

    R = None
    if args.rotation:
        R = np.load(args.rotation).astype(np.float32)
        assert R.shape == (3, 3), f"rotation must be 3x3, got {R.shape}"
        print(f"Loaded rotation matrix from {args.rotation}.")

    # Build label vocabulary from observed modes
    modes = sorted({m for _, m, _ in trials})
    label_map = {m: i for i, m in enumerate(modes)}
    print(f"Label mapping ({len(modes)} classes): {label_map}")

    all_Xw, all_yw = [], []
    per_class_count = {m: 0 for m in modes}
    skipped_short = 0
    for path, mode, subj in trials:
        try:
            X = load_exo_csv(path)
        except Exception as e:
            print(f"  SKIP {subj}/{path.parent.name}: {e}")
            continue
        if R is not None:
            X = apply_rotation(X, R)
        X = downsample(X, args.src_fs, args.tgt_fs)
        Xw, yw = make_windows(X, label_map[mode], window=args.window_size)
        if len(Xw) == 0:
            skipped_short += 1
            continue
        all_Xw.append(Xw); all_yw.append(yw)
        per_class_count[mode] += len(Xw)

    if not all_Xw:
        raise SystemExit("No windows produced. Check downsample/window_size settings.")
    if skipped_short:
        print(f"  ({skipped_short} trials too short for one full window)")

    data  = np.concatenate(all_Xw, axis=0)            # (N, 120, 6)
    label = np.concatenate(all_yw, axis=0)[..., None] # (N, 120, 1)

    print(f"\nFinal shapes: data {data.shape}  label {label.shape}")
    print("Per-class window count:")
    for m in modes:
        print(f"  {m:5s}  id={label_map[m]}  windows={per_class_count[m]}")

    sr_tag = int(args.tgt_fs); wl_tag = int(args.window_size)
    np.save(out_dir / f'data_{sr_tag}_{wl_tag}.npy',  data)
    np.save(out_dir / f'label_{sr_tag}_{wl_tag}.npy', label)
    with open(out_dir / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved:")
    print(f"  {out_dir}/data_{sr_tag}_{wl_tag}.npy")
    print(f"  {out_dir}/label_{sr_tag}_{wl_tag}.npy")
    print(f"  {out_dir}/label_map.json")
    print(f"\nAdd this entry to LIMU-BERT-Public/dataset/data_config.json:")
    entry = {
        f"molinaro_{sr_tag}_{wl_tag}": {
            "sr": sr_tag, "seq_len": wl_tag, "dimension": 6,
            "activity_label_index": 0,
            "activity_label_size":  len(label_map),
            "activity_label":       list(label_map.keys()),
            "size":                 int(data.shape[0]),
        }
    }
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
