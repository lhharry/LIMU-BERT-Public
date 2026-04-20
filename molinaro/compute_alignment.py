"""
Compute a 3x3 rotation matrix R that aligns Molinaro's IMU coordinate frame to your
'self' frame, by matching the gravity vectors estimated from low-motion segments.

The matrix R is then used by preprocess_molinaro.py to rotate every Molinaro acc and
gyro sample into the same frame as your self data.

Usage:
    python compute_alignment.py \\
        --self_csv     path/to/dataset_self.csv \\
        --molinaro_csv path/to/any_molinaro_with_some_still_segment.csv \\
        --out          gravity_R.npy

Tips:
- For best results, point --self_csv and --molinaro_csv at recordings that contain
  several seconds of standing-still / quiet sitting. The script picks the lowest-gyro
  20% of samples as a proxy for "still".
- Gravity gives only 2 degrees of freedom — the residual rotation around the gravity
  axis cannot be recovered from static data. For HAR this is usually fine.
"""

import argparse
import numpy as np
import pandas as pd

SELF_ACC  = ['left_accel_x', 'left_accel_y', 'left_accel_z']
SELF_GYRO = ['left_gyro_x',  'left_gyro_y',  'left_gyro_z']  # deg/s in self
MOLI_ACC  = ['thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l']
MOLI_GYRO = ['thigh_gyro_x_l',  'thigh_gyro_y_l',  'thigh_gyro_z_l']  # rad/s in Molinaro


def estimate_gravity(acc, gyro_rad, quiet_quantile=0.20):
    """Mean acc over the `quiet_quantile` fraction of samples with lowest gyro magnitude."""
    gmag = np.linalg.norm(gyro_rad, axis=1)
    threshold = np.quantile(gmag, quiet_quantile)
    quiet_acc = acc[gmag < threshold]
    return quiet_acc.mean(axis=0)


def rot_a_to_b(a, b):
    """Rodrigues rotation: returns 3x3 R such that R @ a is parallel to b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self_csv',     required=True)
    p.add_argument('--molinaro_csv', required=True)
    p.add_argument('--out',          default='gravity_R.npy')
    p.add_argument('--quantile',     type=float, default=0.20,
                   help='Fraction of lowest-gyro samples treated as "still"')
    args = p.parse_args()

    # self
    self_df = pd.read_csv(args.self_csv, sep=';').dropna()
    self_acc  = self_df[SELF_ACC].values
    self_gyro = self_df[SELF_GYRO].values * np.pi / 180.0   # deg/s -> rad/s
    g_self = estimate_gravity(self_acc, self_gyro, args.quantile)

    # Molinaro
    moli_df = pd.read_csv(args.molinaro_csv, sep=';').dropna()
    moli_acc  = moli_df[MOLI_ACC].values
    moli_gyro = moli_df[MOLI_GYRO].values
    g_moli = estimate_gravity(moli_acc, moli_gyro, args.quantile)

    print(f"Self gravity      : {g_self.round(3)}  (|g|={np.linalg.norm(g_self):.2f})")
    print(f"Molinaro gravity  : {g_moli.round(3)}  (|g|={np.linalg.norm(g_moli):.2f})")
    if abs(np.linalg.norm(g_self) - 9.8) > 2:
        print("  WARNING: |g_self| far from 9.8 — your data may not have enough still segments.")
    if abs(np.linalg.norm(g_moli) - 9.8) > 2:
        print("  WARNING: |g_moli| far from 9.8 — Molinaro reference file may not be still enough.")

    R = rot_a_to_b(g_moli, g_self).astype(np.float32)
    g_moli_rot = R @ g_moli
    print(f"\nRotation matrix R:\n{R.round(4)}")
    print(f"Molinaro gravity AFTER R:  {g_moli_rot.round(3)}  (target: {g_self.round(3)})")
    print(f"Direction error (deg)   :  {np.degrees(np.arccos(np.clip(np.dot(g_moli_rot/np.linalg.norm(g_moli_rot), g_self/np.linalg.norm(g_self)), -1, 1))):.2f}")

    np.save(args.out, R)
    print(f"\nSaved -> {args.out}")
    print("Pass it to preprocess_molinaro.py with --rotation flag.")


if __name__ == '__main__':
    main()
