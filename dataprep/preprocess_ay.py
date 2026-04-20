#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 AY (Camargo et al.) 数据集的 CSV 转成 LIMU-BERT 所需的 .npy 格式
======================================================================
输入目录结构:
  D:/01_Code/DATA/OpenSource/AY_Data/
    AB06/training_data/*.csv
    AB07/training_data/*.csv
    ...
每个 CSV 列: Header, foot_*, shank_*, thigh_*, trunk_* (共24通道), Label

输出:
  dataset/ay/data_20_120.npy   shape: (N, 120, 24)
  dataset/ay/label_20_120.npy  shape: (N, 120, 1)
"""
import os
import glob
import numpy as np
import pandas as pd

# ---------------- 可调参数 ----------------
DATA_ROOT   = r"D:/01_Code/DATA/OpenSource/AY_Data"   # 你的数据根目录
OUT_DIR     = r"./dataset/ay"                         # 输出 npy 的目录
SEQ_LEN     = 120        # 每个样本的时间步数
STRIDE      = 60         # 滑窗步长 (60 = 50% 重叠)
DOWNSAMPLE  = 10         # 200Hz -> 20Hz (LIMU-BERT 推荐 20Hz)
VERSION_TAG = "20_120"   # 文件名后缀: data_{tag}.npy
# -----------------------------------------

# 24 个传感器通道的列名顺序 (foot -> shank -> thigh -> trunk)
FEATURE_COLS = []
for sensor in ["foot", "shank", "thigh", "trunk"]:
    for axis in ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]:
        FEATURE_COLS.append(f"{sensor}_{axis}")

# 标签字符串 -> 整数索引 (索引顺序就是 dataset.json 里 label_names 的顺序)
LABEL_MAP = {
    "idle":       0,
    "stand":      1,
    "walk":       2,
    "stand-walk": 3,
    "walk-stand": 4,
    "turn1":      5,
    "turn2":      6,
}


def load_one_csv(path):
    """读取单个 CSV, 返回 (T, 24) 的 features 和 (T,) 的 label_id."""
    df = pd.read_csv(path)
    # 跳过任何标签不在 LABEL_MAP 里的行
    df = df[df["Label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    if len(df) == 0:
        return None, None
    feats  = df[FEATURE_COLS].values.astype(np.float32)
    labels = df["Label"].map(LABEL_MAP).values.astype(np.int64)
    return feats, labels


def slide_window(feats, labels, seq_len, stride):
    """对一条 trial 切固定长度窗口. 窗口标签取众数."""
    X, Y = [], []
    T = len(feats)
    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        win_x = feats[start:end]                       # (seq_len, 24)
        win_y = labels[start:end]                      # (seq_len,)
        X.append(win_x)
        # LIMU-BERT 期望 label shape = (seq_len, 1), 整段保留逐帧标签
        Y.append(win_y.reshape(-1, 1))
    return X, Y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 找所有 subject 的 training_data 目录
    csv_files = sorted(glob.glob(
        os.path.join(DATA_ROOT, "AB*", "training_data", "*.csv")
    ))
    print(f"[INFO] Found {len(csv_files)} CSV files")

    all_X, all_Y = [], []
    for i, f in enumerate(csv_files):
        feats, labels = load_one_csv(f)
        if feats is None:
            print(f"  [skip] {f} (no valid labels)")
            continue

        # 降采样: 200Hz -> 20Hz
        if DOWNSAMPLE > 1:
            feats  = feats[::DOWNSAMPLE]
            labels = labels[::DOWNSAMPLE]

        if len(feats) < SEQ_LEN:
            continue

        X, Y = slide_window(feats, labels, SEQ_LEN, STRIDE)
        all_X.extend(X)
        all_Y.extend(Y)

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(csv_files)} files, total windows = {len(all_X)}")

    X = np.stack(all_X, axis=0)         # (N, seq_len, 24)
    Y = np.stack(all_Y, axis=0)         # (N, seq_len, 1)
    print(f"[DONE] data shape = {X.shape}, label shape = {Y.shape}")

    # 保存
    np.save(os.path.join(OUT_DIR, f"data_{VERSION_TAG}.npy"),  X)
    np.save(os.path.join(OUT_DIR, f"label_{VERSION_TAG}.npy"), Y)
    print(f"[SAVED] {OUT_DIR}/data_{VERSION_TAG}.npy")
    print(f"[SAVED] {OUT_DIR}/label_{VERSION_TAG}.npy")

    # 打印各类样本数 (按窗口的多数标签统计)
    from collections import Counter
    win_labels = []
    for y in Y:
        vals, cnts = np.unique(y, return_counts=True)
        win_labels.append(int(vals[np.argmax(cnts)]))
    print("[CLASS DIST]", Counter(win_labels))


if __name__ == "__main__":
    main()
