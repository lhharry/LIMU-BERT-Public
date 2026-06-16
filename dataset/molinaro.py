"""
Molinaro hip-exo dataset (04_Monilaro) -> LIMU-BERT NPY format.

Aligned with scherpereel.py / camargo口径:
    - label shape (N, seq_len, 2) = [activity_id, user_id]
    - block-averaging downsample 200 Hz -> 10 Hz (no scipy / anti-alias filter,
      matches the block-averaging used to pretrain the foundation model)
    - --leg left|right|both  (exo.csv has both _l and _r thigh IMU columns)
    - accel already m/s^2, gyro already rad/s  -> NO unit conversion

Segmentation note: Molinaro has NO per-sample activity flag. Each trial FOLDER
is one steady-state locomotion bout; the mode is the first underscore token of
the folder name (e.g. "LG_C0p0_S1p25_UC_1_1" -> "LG"). Transition modes TRA/TRB
are excluded via --include_modes (default: LG RA RD SA SD ST).

7-class space: LG (level ground) is split into walk/jog by the folder's speed
token S<x>p<y> (m/s, 'p' = decimal point; speed > 1.5 -> jog, else walk, matching
scherpereel.py). RA/RD/SA/SD/ST map to rampascent/rampdescent/stairascent/
stairdescent/stand. 'stand' is kept last to align with scherpereel's dense ids.
Overground/standing trials carry speed S0p0 (= 0.0) and so land in walk/stand.



Usage:
    python dataset/molinaro.py                 # both default args, left leg
    python dataset/molinaro.py --leg both
"""

import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW_SR     = 200
TARGET_SR  = 10
SEQ_LEN    = 20

DATASET_PATH = r'D:\01_Code\DATA\OpenSource\04_Monilaro\dataset'
DEFAULT_MODES = ['LG', 'RA', 'RD', 'SA', 'SD', 'ST']   # exclude TRA/TRB transitions

LEFT_COLS  = ['thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l',
              'thigh_gyro_x_l',  'thigh_gyro_y_l',  'thigh_gyro_z_l']
RIGHT_COLS = ['thigh_accel_x_r', 'thigh_accel_y_r', 'thigh_accel_z_r',
              'thigh_gyro_x_r',  'thigh_gyro_y_r',  'thigh_gyro_z_r']

LEG_COLS = {
    'left':  [LEFT_COLS],
    'right': [RIGHT_COLS],
    'both':  [LEFT_COLS, RIGHT_COLS],
}

WALK_JOG_SPEED_THRESHOLD = 1.5   # m/s; speed > threshold -> jog (matches scherpereel.py)

# 7-class space: subset of the shared dense vocabulary (molinaro has no turn /
# sit-stand-transition trials). LG is split into walk/jog by speed. 'stand' is
# kept last to line up with scherpereel's dense ordering.
ACTIVITIES_7 = ["walk", "jog", "rampascent", "rampdescent",
                "stairascent", "stairdescent", "stand"]

MODE_TO_ACTIVITY = {
    'RA': 'rampascent', 'RD': 'rampdescent',
    'SA': 'stairascent', 'SD': 'stairdescent', 'ST': 'stand',
}


def parse_speed(folder_name):
    """Speed token S<x>p<y> (m/s) -> float ('p' is the decimal point).
    'LG_C0p0_S1p25_UC_1_1' -> 1.25.  Overground/standing trials use S0p0 -> 0.0.
    """
    m = re.search(r'(?:^|_)S(\d+)p(\d+)(?:_|$)', folder_name)
    return float(f'{m.group(1)}.{m.group(2)}') if m else 0.0


def folder_to_activity(folder_name, mode):
    """Map a trial folder to one of ACTIVITIES_7. LG -> walk/jog by speed."""
    if mode == 'LG':
        return 'jog' if parse_speed(folder_name) > WALK_JOG_SPEED_THRESHOLD else 'walk'
    return MODE_TO_ACTIVITY.get(mode)


def label_user(name):
    return int(name[2:]) - 10   # 'AB10' -> 0 ... 'AB34' -> 24


def down_sample(data, raw_sr, target_sr):
    """Block-averaging downsample (same as scherpereel.py)."""
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


def discover_trials(root_dir, include_modes):
    """Walk <root>/<AB##>/<Trial>/exo.csv. Returns list of (exo_path, mode, subject)."""
    out = []
    root = Path(root_dir)
    for subj in sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith('AB')):
        for trial in sorted(t for t in subj.iterdir() if t.is_dir()):
            mode = trial.name.split('_')[0]
            if include_modes and mode not in include_modes:
                continue
            exo = trial / 'exo.csv'
            if exo.exists():
                out.append((exo, mode, subj.name))
    return out


def load_sensor_data(trials, act_index, leg, seq_len, raw_sr, target_sr):
    data_all, label_all = [], []
    skipped_short = 0

    for path, mode, subj in trials:
        act = folder_to_activity(path.parent.name, mode)
        if act is None:
            continue
        label_act = act_index[act]
        label_u   = label_user(subj)
        df = pd.read_csv(path)

        for cols in LEG_COLS[leg]:
            if any(c not in df.columns for c in cols):
                continue

            sensor = df[cols].values.astype(float)
            finite = np.all(np.isfinite(sensor), axis=1)
            sensor = sensor[finite]
            if len(sensor) == 0:
                continue
            # accel m/s^2, gyro rad/s already -> no conversion

            sensor_down = down_sample(sensor, raw_sr, target_sr)
            n_windows = sensor_down.shape[0] // seq_len
            if n_windows == 0:
                skipped_short += 1
                continue
            sensor_down = sensor_down[:n_windows * seq_len].reshape(n_windows, seq_len, 6)

            lbl = np.full((n_windows, seq_len, 2), [[label_act, label_u]], dtype=float)
            data_all.append(sensor_down)
            label_all.append(lbl)

    if skipped_short:
        print(f'  ({skipped_short} trial-legs too short for one full window)')
    return data_all, label_all


def preprocess(path, path_save, version, leg, include_modes,
               raw_sr=RAW_SR, target_sr=TARGET_SR, seq_len=SEQ_LEN):
    trials = discover_trials(path, include_modes)
    if not trials:
        raise SystemExit(f'No trials found under {path} matching modes {include_modes}.')
    subjects  = sorted({s for _, _, s in trials})
    act_index = {a: i for i, a in enumerate(ACTIVITIES_7)}
    print(f'Found {len(trials)} trials, {len(subjects)} subjects.')
    print(f'Label map ({len(ACTIVITIES_7)} classes): {act_index}')

    data_list, label_list = load_sensor_data(trials, act_index, leg, seq_len, raw_sr, target_sr)
    data  = np.concatenate(data_list,  0).astype(np.float32)
    label = np.concatenate(label_list, 0).astype(np.float32)

    ids, counts = np.unique(label[:, 0, 0].astype(int), return_counts=True)
    dist = {ACTIVITIES_7[i]: int(c) for i, c in zip(ids, counts)}
    print(f'All data processed [{leg}]. data={data.shape}  label={label.shape}')
    print(f'Per-class windows: {dist}')
    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data_'  + version + '.npy'), data)
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), label)
    with open(os.path.join(path_save, 'label_map.json'), 'w') as f:
        json.dump(act_index, f, indent=2)

    key = f'molinaro_{version}'
    entry = {key: {
        'sr': target_sr, 'seq_len': seq_len, 'dimension': 6,
        'activity_label_index': 0, 'activity_label_size': len(ACTIVITIES_7),
        'activity_label': list(ACTIVITIES_7),
        'user_label_index': 1, 'user_label_size': len(subjects),
        'size': int(data.shape[0]),
    }}
    print('\nAdd to data_config.json:')
    print(json.dumps(entry, indent=4))
    return data, label


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir',     default=DATASET_PATH)
    p.add_argument('--leg',           choices=['left', 'right', 'both'], default='both')
    p.add_argument('--include_modes', nargs='*', default=DEFAULT_MODES)
    p.add_argument('--tgt_sr',        type=int, default=TARGET_SR)
    p.add_argument('--seq_len',       type=int, default=SEQ_LEN)
    args = p.parse_args()

    suffix  = '' if args.leg == 'left' else f'_{args.leg}'
    version = f'{args.tgt_sr}_{args.seq_len}{suffix}_dense_7cls'

    preprocess(args.input_dir, 'dataset/molinaro', version, args.leg, args.include_modes,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
