#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/04/23
# @File    : camargo.py
# @Description : http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al
'''
Dense-label scheme: transition classes are absorbed into the *later* activity
of the transition pair (e.g. "stand-walk" -> "walk", "walk-rampascent" ->
"rampascent", "rampascent-walk" -> "walk"). Every timestep therefore carries
a real activity label.

Final label set (9 classes):
  0: stand
  1: walk
  2: turn
  3: jog
  4: rampascent
  5: rampdescent
  6: stairascent
  7: stairdescent
'''

import os
import numpy as np
import pandas as pd

RAW_SR = 200
DATASET_PATH = r'D:\01_Code\DATA\OpenSource\AY_Camargo'

# Compact, dense activity set (no transition classes)
ACTIVITY_NAMES = ["stand", "walk", "turn", "jog",
                  "rampascent", "rampdescent",
                  "stairascent", "stairdescent"]

# Map raw CSV labels (which include transition pairs "X-Y") to a dense
# activity. For transitions, we keep the *later* activity (Y).
RAW_TO_DENSE = {
    "stand":              "stand",
    "stand-walk":         "walk",
    "walk":               "walk",
    "walk-stand":         "stand",
    "turn1":              "turn",
    "turn2":              "turn",
    "jog":                "jog",
    "rampascent":         "rampascent",
    "walk-rampascent":    "rampascent",
    "rampascent-walk":    "walk",
    "rampdescent":        "rampdescent",
    "walk-rampdescent":   "rampdescent",
    "rampdescent-walk":   "walk",
    "stairascent":        "stairascent",
    "walk-stairascent":   "stairascent",
    "stairascent-walk":   "walk",
    "stairdescent":       "stairdescent",
    "walk-stairdescent":  "stairdescent",
    "stairdescent-walk":  "walk",
}

SENSOR_COLS = ['thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z',
               'thigh_Gyro_X', 'thigh_Gyro_Y', 'thigh_Gyro_Z']


def label_activity(name):
    dense_name = RAW_TO_DENSE.get(name)
    if dense_name is None:
        return -1
    return ACTIVITY_NAMES.index(dense_name)


def label_user(name):
    # folder name like 'AB06' -> user id 5 (keeps original numbering, with gaps)
    return int(name[2:]) - 1


def down_sample(data, raw_sr, target_sr):
    window_sample = raw_sr * 1.0 / target_sr
    result = []
    if window_sample < 1:
        raise ValueError('target_sr must be less than or equal to raw_sr')
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 < data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                i += window
    return np.array(result)


def load_sensor_data(path, seq_len, raw_sr, target_sr):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if not dir.startswith('AB'):
                continue
            label_u = label_user(dir)
            path_sub = os.path.join(root, dir, 'training_data')
            if not os.path.isdir(path_sub):
                continue
            for name in sorted(os.listdir(path_sub)):
                if not name.endswith('.csv'):
                    continue
                path_exp = os.path.join(path_sub, name)
                df = pd.read_csv(path_exp)
                sensor = df[SENSOR_COLS].values
                acts_raw = df['Label'].values

                # Remap raw labels (incl. transitions) to dense indices FIRST,
                # then segment by contiguous dense label. This way adjacent
                # segments that collapse to the same activity (e.g. a
                # "walk-stand" transition immediately followed by "stand")
                # merge into one longer segment instead of being cut.
                dense = np.array([label_activity(a) for a in acts_raw])

                i = 0
                while i < len(dense):
                    label_act = dense[i]
                    j = i
                    while j < len(dense) and dense[j] == label_act:
                        j += 1
                    if label_act < 0:
                        i = j
                        continue
                    sensor_down = down_sample(sensor[i:j, :], raw_sr, target_sr)
                    if sensor_down.shape[0] > seq_len:
                        sensor_down = sensor_down[:sensor_down.shape[0] // seq_len * seq_len, :]
                        sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
                        sensor_label = np.ones((sensor_down.shape[0], sensor_down.shape[1], 1))
                        sensor_label = np.concatenate([sensor_label * label_act, sensor_label * label_u], 2)
                        data.append(sensor_down)
                        label.append(sensor_label)
                    i = j
        break  # only walk top-level subject folders
    return data, label


def preprocess(path, path_save, version, raw_sr=120, target_sr=10, seq_len=120):
    data, label = load_sensor_data(path, seq_len, raw_sr, target_sr)
    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    print('All data processed. Size: %d' % (data.shape[0]))
    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data_' + version + '.npy'), np.array(data))
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), np.array(label))
    return data, label


path_save = r'camargo'
version = r'10_20_dense_8cls'
data, label = preprocess(DATASET_PATH, path_save, version, raw_sr=RAW_SR, target_sr=10, seq_len=20)