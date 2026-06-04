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

LEG_COLS = {
    'left':  [LEFT_ACCEL  + LEFT_GYRO],
    'right': [RIGHT_ACCEL + RIGHT_GYRO],
    'both':  [LEFT_ACCEL  + LEFT_GYRO, RIGHT_ACCEL + RIGHT_GYRO],
}

RAD_PER_DEG = 1.0 / 57.29578


def get_base_activity(folder_name):
    """'ball_toss_1_2_center_off' -> 'ball_toss'  (before first digit token)"""
    parts = folder_name.split('_')
    for i, p in enumerate(parts):
        if p.isdigit():
            return '_'.join(parts[:i])
    return folder_name


def build_label_map(root):
    activities = set()
    for subj in os.listdir(root):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path) or not subj.startswith('BT'):
            continue
        for folder in os.listdir(subj_path):
            if os.path.isdir(os.path.join(subj_path, folder)):
                activities.add(get_base_activity(folder))
    return {act: i for i, act in enumerate(sorted(activities))}


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


def load_sensor_data(path, label_map, seq_len, raw_sr, target_sr, col_sets):
    data_all, label_all = [], []

    for subj in sorted(os.listdir(path)):
        subj_path = os.path.join(path, subj)
        if not os.path.isdir(subj_path) or not subj.startswith('BT'):
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

            exo_file = os.path.join(folder_path, folder + '_exo.csv')
            if not os.path.exists(exo_file):
                candidates = [f for f in os.listdir(folder_path) if f.endswith('_exo.csv')]
                if not candidates:
                    continue
                exo_file = os.path.join(folder_path, candidates[0])

            df = pd.read_csv(exo_file)

            for cols in col_sets:
                if any(c not in df.columns for c in cols):
                    continue

                sensor = df[cols].values.astype(float)
                finite = np.all(np.isfinite(sensor), axis=1)
                sensor = sensor[finite]
                if len(sensor) == 0:
                    continue

                sensor[:, 3:] *= RAD_PER_DEG  # gyro deg/s -> rad/s

                sensor_down = down_sample(sensor, raw_sr, target_sr)
                n_windows = sensor_down.shape[0] // seq_len
                if n_windows == 0:
                    continue
                sensor_down = sensor_down[:n_windows * seq_len].reshape(n_windows, seq_len, 6)

                lbl = np.full((n_windows, seq_len, 2), [[label_act, label_u]], dtype=float)
                data_all.append(sensor_down)
                label_all.append(lbl)

    return data_all, label_all


def preprocess(path, path_save, version, leg, raw_sr=RAW_SR, target_sr=TARGET_SR, seq_len=SEQ_LEN):
    label_map = build_label_map(path)
    print(f'Label map ({len(label_map)} classes):', label_map)

    data_list, label_list = load_sensor_data(
        path, label_map, seq_len, raw_sr, target_sr, LEG_COLS[leg])
    data  = np.concatenate(data_list,  0).astype(np.float32)
    label = np.concatenate(label_list, 0).astype(np.float32)

    print(f'All data processed [{leg}]. data={data.shape}  label={label.shape}')
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
