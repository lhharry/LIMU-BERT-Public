#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : scherpereel_exo.py
# @Description : 03_MonilaroScherpereel dataset -> LiMU-BERT NPY format.
# Same 26-activity protocol as scherpereel.py but collected with a hip exoskeleton.
# Output: data_<version>.npy  (N, 20, 6)  float32
#         label_<version>.npy (N, 20, 2)  float  [activity_id, user_id]
#
# Usage:
#   python scherpereel_exo.py                    # left leg, 10 Hz, seq_len=20
#   python scherpereel_exo.py --leg right
#   python scherpereel_exo.py --leg both

import os
import json
import argparse
import numpy as np
import pandas as pd

RAW_SR     = 200
TARGET_SR  = 10
SEQ_LEN    = 20
N_SUBJECTS = 17   # BT01-BT17

DATASET_PATH = r'D:\01_Code\DATA\OpenSource\03_MonilaroScherpereel\Phase1And2_Parsed'

LEFT_ACCEL  = ['thigh_imu_l_accel_x', 'thigh_imu_l_accel_y', 'thigh_imu_l_accel_z']
LEFT_GYRO   = ['thigh_imu_l_gyro_x',  'thigh_imu_l_gyro_y',  'thigh_imu_l_gyro_z']
RIGHT_ACCEL = ['thigh_imu_r_accel_x', 'thigh_imu_r_accel_y', 'thigh_imu_r_accel_z']
RIGHT_GYRO  = ['thigh_imu_r_gyro_x',  'thigh_imu_r_gyro_y',  'thigh_imu_r_gyro_z']

# leg -> list of (sensor_cols, activity_flag column). Mirrors scherpereel.py:
# each trial ships an <name>_activity_flag.csv (time,left,right) row-aligned to
# the exo csv; only rows where the per-leg flag == 1 are the subject actually
# performing the labeled activity (the rest are setup / transition / rest).
LEG_SPEC = {
    'left':  [(LEFT_ACCEL  + LEFT_GYRO,  'left')],
    'right': [(RIGHT_ACCEL + RIGHT_GYRO, 'right')],
    'both':  [(LEFT_ACCEL  + LEFT_GYRO,  'left'),
              (RIGHT_ACCEL + RIGHT_GYRO, 'right')],
}

RAD_PER_DEG = 1.0 / 57.29578

DENSE_ACTIVITIES = ["stand", "walk", "turn", "jog",
                    "rampascent", "rampdescent",
                    "stairascent", "stairdescent", "sit-stand-transition"]

WALK_JOG_SPEED_THRESHOLD = 1.5

def reconstruct_label(folder_name):
    """Drop pure-integer trial/step index tokens, keep the rest.
    'incline_walk_1_down5'      -> 'incline_walk_down5'
    'stairs_1_10_down'          -> 'stairs_down'
    'sit_to_stand_1_2_short-arm'-> 'sit_to_stand_short-arm'
    'normal_walk_1_0-6'         -> 'normal_walk_0-6'  ('-' is a decimal point)
    """
    return '_'.join(p for p in folder_name.split('_') if not p.isdigit())


def dense_label(folder_name):
    """Map a raw trial folder name to one of DENSE_ACTIVITIES, or None to drop."""
    name = reconstruct_label(folder_name)
    if name.startswith('incline_walk'):
        if 'up' in name:   return 'rampascent'
        if 'down' in name: return 'rampdescent'
        return None
    if name.startswith('normal_walk'):
        toks = name.split('_')
        if 'shuffle' in toks or 'skip' in toks:   # not steady-speed walking -> drop
            return None
        for t in toks:                            # find the speed token, e.g. '1-2'
            try:                                  # ('-' is a decimal point -> 1.2)
                speed = float(t.replace('-', '.'))
            except ValueError:
                continue                          # skips 'normal','walk','on'/'off'/'hilo'
            return 'jog' if speed > WALK_JOG_SPEED_THRESHOLD else 'walk'
        return None
    if name.startswith('sit_to_stand'):
        return 'sit-stand-transition'
    if name.startswith('stairs'):
        if 'up' in name:   return 'stairascent'
        if 'down' in name: return 'stairdescent'
        return None
    if name.startswith('turn_and_step'):
        return 'turn'
    return None


def build_label_map():
    return {name: i for i, name in enumerate(DENSE_ACTIVITIES)}


def label_user(name):
    return int(name[2:]) - 1   # 'BT01' -> 0


def down_sample(data, raw_sr, target_sr):
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


def load_sensor_data(path, label_map, seq_len, raw_sr, target_sr, leg):
    data_all, label_all = [], []
    n_no_flag = 0

    for subj in sorted(os.listdir(path)):
        subj_path = os.path.join(path, subj)
        if not os.path.isdir(subj_path) or not subj.startswith('BT'):
            continue
        label_u = label_user(subj)

        for folder in sorted(os.listdir(subj_path)):
            folder_path = os.path.join(subj_path, folder)
            if not os.path.isdir(folder_path):
                continue
            dense = dense_label(folder)
            if dense is None:
                continue
            label_act = label_map[dense]

            exo_file = os.path.join(folder_path, folder + '_exo.csv')
            if not os.path.exists(exo_file):
                candidates = [f for f in os.listdir(folder_path) if f.endswith('_exo.csv')]
                if not candidates:
                    continue
                exo_file = os.path.join(folder_path, candidates[0])
            df = pd.read_csv(exo_file)

            flag_files = [f for f in os.listdir(folder_path) if f.endswith('_activity_flag.csv')]
            if not flag_files:
                n_no_flag += 1
                continue
            flag_df = pd.read_csv(os.path.join(folder_path, flag_files[0]))

            # align exo and flag row-for-row (same 200 Hz time base)
            m = min(len(df), len(flag_df))
            df, flag_df = df.iloc[:m], flag_df.iloc[:m]

            for cols, flag_col in LEG_SPEC[leg]:
                if any(c not in df.columns for c in cols) or flag_col not in flag_df.columns:
                    continue

                sensor = df[cols].values.astype(float)
                flag   = (flag_df[flag_col].values == 1)

                # Keep ALL flag==1 rows of this trial (one activity), concatenated
                # in time order, then window (same as scherpereel.py).
                keep = flag & np.all(np.isfinite(sensor), axis=1)
                seg = sensor[keep]
                if len(seg) < seq_len * (raw_sr // target_sr):
                    continue  # <1 window worth of active data in whole trial

                seg = seg.copy()
                seg[:, 3:] *= RAD_PER_DEG  # gyro deg/s -> rad/s

                seg_down = down_sample(seg, raw_sr, target_sr)
                n_windows = seg_down.shape[0] // seq_len
                if n_windows == 0:
                    continue
                seg_down = seg_down[:n_windows * seq_len].reshape(n_windows, seq_len, 6)

                lbl = np.full((n_windows, seq_len, 2), [[label_act, label_u]], dtype=float)
                data_all.append(seg_down)
                label_all.append(lbl)

    if n_no_flag:
        print(f'WARNING: {n_no_flag} trials had no activity_flag.csv and were skipped.')
    return data_all, label_all


def preprocess(path, path_save, version, leg, raw_sr=RAW_SR, target_sr=TARGET_SR, seq_len=SEQ_LEN):
    label_map = build_label_map()
    print(f'Label map ({len(label_map)} classes):', label_map)

    data_list, label_list = load_sensor_data(
        path, label_map, seq_len, raw_sr, target_sr, leg)
    data  = np.concatenate(data_list,  0).astype(np.float32)
    label = np.concatenate(label_list, 0).astype(np.float32)

    ids, counts = np.unique(label[:, 0, 0].astype(int), return_counts=True)
    dist = {DENSE_ACTIVITIES[i]: int(c) for i, c in zip(ids, counts)}
    print(f'All data processed [{leg}, flag==1 only]. data={data.shape}  label={label.shape}')
    print(f'Per-class windows: {dist}')
    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data_'  + version + '.npy'), data)
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), label)
    
    key = f'scherpereel_exo_{version}'
    entry = {key: {
        'sr': target_sr, 'seq_len': seq_len, 'dimension': 6,
        'activity_label_index': 0, 'activity_label_size': len(label_map),
        'activity_label': list(label_map.keys()),
        'user_label_index': 1, 'user_label_size': N_SUBJECTS,
        'size': int(data.shape[0]),
    }}
    print('\nAdd to data_config.json:')
    print(json.dumps(entry, indent=4))
    return data, label


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--leg',     choices=['left', 'right', 'both'], default='left')
    p.add_argument('--tgt_sr',  type=int, default=TARGET_SR)
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    args = p.parse_args()

    suffix  = '' if args.leg == 'left' else f'_{args.leg}'
    version = f'{args.tgt_sr}_{args.seq_len}{suffix}'

    preprocess(DATASET_PATH, 'dataset/scherpereel_exo', version, args.leg,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
