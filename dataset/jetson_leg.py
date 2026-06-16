"""
Jetson bilateral leg IMU recordings -> LIMU-BERT NPY format.

Source
------
DATA/jetson/**/<HH_MM_SS>_<class>_leg[/...]  (incl. DATA/jetson/stair/<...>).
Each trial folder has:
    accelerometers.csv  Time, Left_{x,y,z}, Right_{x,y,z}   (m/s^2)
    gyroscopes.csv      Time, Left_{x,y,z}, Right_{x,y,z}   (rad/s)
    label.csv           Time, Label                         (per-sample)
label.csv was produced by jetson/make_labels.py and has a
stand -> activity -> stand structure with timestamps identical to the sensors.

Conventions (aligned with molinaro.py / camargo_v2.py / jetson_compare.py)
-------------------------------------------------------------------------
* 7 dense activity classes present in the jetson leg data (no rampdescent):
  ["stand", "walk", "turn", "jog", "rampascent", "stairascent", "stairdescent"]
  -> compact ids 0..6. Raw label tokens are normalised via LABEL_TO_DENSE
  (only "rampup" -> "rampascent"; the rest already match).
* Axis remap to the camargo convention: jetson (y, x, z) -> camargo (x, y, z),
  applied to both accel and gyro. Identical to load_jetson_trial in
  jetson_compare.py, so columns line up with a camargo-pretrained model.
* Units already match the camargo/molinaro NPYs (accel m/s^2, gyro rad/s),
  so NO unit conversion.
* Block-averaging downsample ~63 Hz -> 10 Hz (no anti-alias filter, matching
  how the foundation model was pretrained). Each contiguous activity segment is
  downsampled and windowed independently so no window straddles a label change.
* --leg left|right|both: each chosen leg becomes its own 6-dim sample stream
  (both -> Left and Right pooled as separate samples; version suffix _both).
* label shape (N, seq_len, 2) = [activity_id, user_id]. A single subject/session
  is assumed, so user_id is always 0 (user_label_size = 1).
* "_zeroed" (gravity-removed) variants are skipped to keep the accel
  distribution consistent with the raw-gravity recordings.

Usage (run from the LIMU-BERT-Public repo root):
    python dataset/jetson_leg.py                 # both legs -> version 10_20_both
    python dataset/jetson_leg.py --leg left      # left leg  -> version 10_20
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

RAW_SR    = 63    # measured median jetson sampling rate (~62-66 Hz across trials)
TARGET_SR = 10
SEQ_LEN   = 20

DATASET_PATH = r'D:\01_Code\DATA\jetson'

# Dense 7-class label space (compact ids 0..6).
ACTIVITY_NAMES = ["stand", "walk", "turn", "jog",
                  "rampascent", "stairascent", "stairdescent"]

# Raw label.csv token -> dense activity name.
LABEL_TO_DENSE = {
    "stand": "stand", "walk": "walk", "turn": "turn", "jog": "jog",
    "rampup": "rampascent",
    "stairascent": "stairascent", "stairdescent": "stairdescent",
}

# Per-side source columns in camargo axis order (jetson y->x, x->y, z->z),
# accel first then gyro. Matches jetson_compare.load_jetson_trial.
def side_cols(side):
    return [f"{side}_y", f"{side}_x", f"{side}_z"]

LEG_SIDES = {'left': ['Left'], 'right': ['Right'], 'both': ['Left', 'Right']}


def down_sample(data, raw_sr, target_sr):
    """Block-averaging downsample (same routine as molinaro.py)."""
    window_sample = raw_sr * 1.0 / target_sr
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            result.append(np.mean(data[i:i + window], 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while i + window + 1 < data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                result.append(np.mean(data[i:i + window + 1], 0))
                i += window + 1
            else:
                result.append(np.mean(data[i:i + window], 0))
                i += window
    return np.array(result)


def discover_trials(root):
    """Yield (folder, accel_path, gyro_path, label_path) for every leg trial."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        fset = set(files)
        if 'label.csv' not in fset or 'gyroscopes.csv' not in fset:
            continue
        accel = glob.glob(os.path.join(dirpath, 'accelerometers*.csv'))
        if not accel:
            continue
        toks = os.path.basename(dirpath).split('_')
        if len(toks) < 5 or toks[4].lower() != 'leg':   # position must be leg
            continue
        if 'zeroed' in toks[5:]:                          # gravity-removed variant
            print(f'  [skip] zeroed variant: {os.path.basename(dirpath)}')
            continue
        out.append((dirpath, accel[0],
                    os.path.join(dirpath, 'gyroscopes.csv'),
                    os.path.join(dirpath, 'label.csv')))
    return out


def load_trial_sides(accel_path, gyro_path, label_path, sides):
    """
    Return (dense_labels[list], {side: (N,6) float array}) for one trial,
    truncated to the common length of the three files.
    """
    acc = pd.read_csv(accel_path)
    gyr = pd.read_csv(gyro_path)
    lab = pd.read_csv(label_path)
    n = min(len(acc), len(gyr), len(lab))
    acc, gyr, lab = acc.iloc[:n], gyr.iloc[:n], lab.iloc[:n]

    dense = [LABEL_TO_DENSE.get(x) for x in lab['Label']]
    if any(d is None for d in dense):
        bad = sorted({x for x, d in zip(lab['Label'], dense) if d is None})
        raise ValueError(f'unknown label token(s) {bad} in {label_path}')

    sensors = {}
    for side in sides:
        cols = side_cols(side)
        a = acc[cols].to_numpy(dtype=float)
        g = gyr[cols].to_numpy(dtype=float)
        sensors[side] = np.hstack([a, g])     # (N,6): accel xyz, gyro xyz
    return dense, sensors


def segment_and_window(dense, sensor, seq_len, raw_sr, target_sr):
    """
    Split one (N,6) stream into fixed-length windows per contiguous activity
    segment. Returns (data_list, label_act_list); user id is added by caller.
    """
    data, acts = [], []
    i = 0
    while i < len(dense):
        j = i
        while j < len(dense) and dense[j] == dense[i]:
            j += 1
        seg = sensor[i:j]
        if np.all(np.isfinite(seg)):
            down = down_sample(seg, raw_sr, target_sr)
            n_win = down.shape[0] // seq_len
            if n_win:
                down = down[:n_win * seq_len].reshape(n_win, seq_len, 6)
                data.append(down)
                acts.append(np.full(n_win, ACTIVITY_NAMES.index(dense[i])))
        i = j
    return data, acts


def preprocess(path, path_save, version, leg,
               raw_sr=RAW_SR, target_sr=TARGET_SR, seq_len=SEQ_LEN):
    trials = discover_trials(path)
    if not trials:
        raise SystemExit(f'No leg trials with label.csv found under {path}.')
    sides = LEG_SIDES[leg]

    data_all, label_all = [], []
    for folder, accel_path, gyro_path, label_path in sorted(trials):
        dense, sensors = load_trial_sides(accel_path, gyro_path, label_path, sides)
        n_win = 0
        for side in sides:
            data, acts = segment_and_window(dense, sensors[side],
                                            seq_len, raw_sr, target_sr)
            for d, a in zip(data, acts):
                lbl = np.zeros((d.shape[0], seq_len, 2))     # user id 0
                lbl[:, :, 0] = a[:, None]
                data_all.append(d)
                label_all.append(lbl)
                n_win += d.shape[0]
        print(f'  {os.path.basename(folder):28s} -> {n_win} windows')

    data  = np.concatenate(data_all, 0).astype(np.float32)
    label = np.concatenate(label_all, 0).astype(np.float32)

    # Per-class window counts (over all emitted samples).
    ids, counts = np.unique(label[:, 0, 0].astype(int), return_counts=True)
    dist = {ACTIVITY_NAMES[i]: int(c) for i, c in zip(ids, counts)}

    print('\nAll data processed. Size: %d' % (data.shape[0]))
    print(f'[{leg}] data={data.shape}  label={label.shape}')
    print(f'Per-class windows: {dist}')

    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data_'  + version + '.npy'), data)
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), label)
    label_map = {name: i for i, name in enumerate(ACTIVITY_NAMES)}
    with open(os.path.join(path_save, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    entry = {f'jetson_leg_{version}': {
        'sr': target_sr, 'seq_len': seq_len, 'dimension': 6,
        'activity_label_index': 0, 'activity_label_size': len(ACTIVITY_NAMES),
        'activity_label': ACTIVITY_NAMES,
        'user_label_index': 1, 'user_label_size': 1,
        'model_label_index': -1, 'model_label_size': 0,
        'size': int(data.shape[0]),
    }}
    print('\nAdd to dataset/data_config.json:')
    print(json.dumps(entry, indent=4))
    return data, label


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default=DATASET_PATH)
    p.add_argument('--leg', choices=['left', 'right', 'both'], default='both')
    p.add_argument('--tgt_sr', type=int, default=TARGET_SR)
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    args = p.parse_args()

    suffix  = '' if args.leg == 'left' else f'_{args.leg}'
    version = f'{args.tgt_sr}_{args.seq_len}{suffix}'

    preprocess(args.input_dir, 'dataset/jetson_leg', version, args.leg,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
