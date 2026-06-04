#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : scherpereel.py
# @Description : https://doi.org/10.1038/s41597-023-02660-6
# Preprocesses 02_Scherpereel dataset into LiMU-BERT NPY format.
# Output: data_<version>.npy  (N, 20, 6)  float32
#         label_<version>.npy (N, 20, 2)  float  [activity_id, user_id]
#
# IMPORTANT: each trial ships an <name>_activity_flag.csv with per-sample,
# per-leg flags (columns: time,left,right). Only rows where the flag == 1 are
# the subject actually performing the labeled activity; the rest are setup /
# transition / rest. We keep ONLY flagged rows, and window WITHIN each
# contiguous flagged segment so windows never bridge a rest gap.
#
# Usage:
#   python scherpereel.py                    # left leg, 10 Hz, seq_len=20
#   python scherpereel.py --leg right
#   python scherpereel.py --leg both

import os
import json
import argparse
import numpy as np
import pandas as pd

RAW_SR     = 200
TARGET_SR  = 10
SEQ_LEN    = 20
N_SUBJECTS = 12

DATASET_PATH = r'D:\01_Code\DATA\OpenSource\02_Scherpereel\ProcessedData'

LEFT_ACCEL  = ['LAThigh_ACCX',  'LAThigh_ACCY',  'LAThigh_ACCZ']
LEFT_GYRO   = ['LAThigh_GYROX', 'LAThigh_GYROY', 'LAThigh_GYROZ']
RIGHT_ACCEL = ['RAThigh_ACCX',  'RAThigh_ACCY',  'RAThigh_ACCZ']
RIGHT_GYRO  = ['RAThigh_GYROX', 'RAThigh_GYROY', 'RAThigh_GYROZ']

# leg -> list of (sensor_cols, flag_column_name)
LEG_SPEC = {
    'left':  [(LEFT_ACCEL  + LEFT_GYRO,  'left')],
    'right': [(RIGHT_ACCEL + RIGHT_GYRO, 'right')],
    'both':  [(LEFT_ACCEL  + LEFT_GYRO,  'left'),
              (RIGHT_ACCEL + RIGHT_GYRO, 'right')],
}

RAD_PER_DEG = 1.0 / 57.29578


def get_base_activity(folder_name):
    """'ball_toss_1_center' -> 'ball_toss'  (everything before first digit token)"""
    parts = folder_name.split('_')
    for i, p in enumerate(parts):
        if p.isdigit():
            return '_'.join(parts[:i])
    return folder_name


def build_label_map(root):
    activities = set()
    for subj in os.listdir(root):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path) or not subj.startswith('AB'):
            continue
        for folder in os.listdir(subj_path):
            if os.path.isdir(os.path.join(subj_path, folder)):
                activities.add(get_base_activity(folder))
    return {act: i for i, act in enumerate(sorted(activities))}


def label_user(name):
    return int(name[2:]) - 1   # 'AB01' -> 0


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
        if not os.path.isdir(subj_path) or not subj.startswith('AB'):
            continue
        label_u = label_user(subj)

        for folder in sorted(os.listdir(subj_path)):
            folder_path = os.path.join(subj_path, folder)
            if not os.path.isdir(folder_path):
                continue
            base_act = get_base_activity(folder)
            if base_act not in label_map:
                continue
            label_act = label_map[base_act]

            imu_files = [f for f in os.listdir(folder_path) if f.endswith('_imu_real.csv')]
            if not imu_files:
                continue
            df = pd.read_csv(os.path.join(folder_path, imu_files[0]))

            flag_files = [f for f in os.listdir(folder_path) if f.endswith('_activity_flag.csv')]
            if not flag_files:
                n_no_flag += 1
                continue
            flag_df = pd.read_csv(os.path.join(folder_path, flag_files[0]))

            # align imu and flag row-for-row (same 200 Hz time base)
            m = min(len(df), len(flag_df))
            df, flag_df = df.iloc[:m], flag_df.iloc[:m]

            for cols, flag_col in LEG_SPEC[leg]:
                if any(c not in df.columns for c in cols) or flag_col not in flag_df.columns:
                    continue

                sensor = df[cols].values.astype(float)
                flag   = (flag_df[flag_col].values == 1)

                # Keep ALL flag==1 rows of this trial (one activity), concatenated
                # in time order, then window. Activities like side_shuffle /
                # step_ups consist of many short (<1 s) reps; per-segment
                # windowing would drop every rep and lose the whole class. All
                # kept rows are the same activity, so a window that bridges two
                # reps is still a valid example of it.
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
    label_map = build_label_map(path)
    print(f'Label map ({len(label_map)} classes):', label_map)

    data_list, label_list = load_sensor_data(path, label_map, seq_len, raw_sr, target_sr, leg)
    data  = np.concatenate(data_list,  0).astype(np.float32)
    label = np.concatenate(label_list, 0).astype(np.float32)

    print(f'All data processed [{leg}, flag==1 only]. data={data.shape}  label={label.shape}')
    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data_'  + version + '.npy'), data)
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), label)

    key = f'scherpereel_{version}'
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

    preprocess(DATASET_PATH, 'dataset/scherpereel', version, args.leg,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
