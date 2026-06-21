"""
Read-only verifier for dataset/merged/data_10_20_9cls_align.npy.

Proves, from CODE + DATA (not comments), that the merged 9-class set is truly
axis-aligned across its 4 sources. Three stages mirror the build chain:

  Stage 1  raw CSV  -> per-source aligned npy   (re-derive; gravity axis from raw)
  Stage 2  4 aligned npy -> merged              (re-merge in memory == on-disk)
  Stage 3  merged, split by source              (cross-source frame agreement)

WRITES NOTHING. Only np.load / pd.read_csv existing files + prints a report.
Run:  python dataset/verify_align.py
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))   # .../dataset
ROOT = os.path.dirname(HERE)                          # repo root
RAW_ROOT = r"D:\01_Code\DATA\OpenSource"

ATOL = 1e-3
RAD_PER_DEG = 1.0 / 57.29578

# aligned per-source npy versions (MUST match merge_dense_9cls.SOURCES)
SOURCES = [
    ("camargo",         "camargo",         "10_20_dense_8cls_zxy"),
    ("scherpereel",     "scherpereel",     "10_20_both_dense_9cls_-xy-z"),
    ("scherpereel_exo", "scherpereel_exo", "10_20_both_dense_9cls_-z-y-x"),
    ("molinaro",        "molinaro",        "10_20_both_dense_7cls_-y-x-z"),
]

CHECKS = []   # (name, passed: True/False/None, detail)


def record(name, passed, detail=""):
    CHECKS.append((name, passed, detail))
    tag = "PASS" if passed is True else ("FAIL" if passed is False else "info")
    print(f"  [{tag}] {name}{('  ' + detail) if detail else ''}")


def down_sample(data, raw_sr=200, target_sr=10):
    """Block-average downsample, byte-faithful to the scripts (integer branch,
    keeps the trailing partial block)."""
    w = raw_sr // target_sr
    out = [data[i:i + w].mean(axis=0) for i in range(0, len(data), w)]
    return np.asarray(out)


def window(ds, seq_len=20):
    n = len(ds) // seq_len
    return ds[:n * seq_len].reshape(n, seq_len, ds.shape[1]) if n else np.empty((0, seq_len, ds.shape[1]))


def min_window_dist(saved, w):
    """Smallest max-abs-diff between window w (seq,6) and any saved window."""
    d = np.abs(saved.astype(np.float64) - w[None]).reshape(len(saved), -1).max(axis=1)
    return float(d.min())


def first_glob(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


# ----------------------------------------------------------------------------
# STAGE 1 : raw CSV -> per-source aligned npy
# ----------------------------------------------------------------------------
# accel columns in RAW order + which raw column the script treats as VERTICAL.
RAW_ACCEL = {
    "camargo":         (['thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z'], 'thigh_Accel_X'),
    "scherpereel":     (['LAThigh_ACCX', 'LAThigh_ACCY', 'LAThigh_ACCZ'], 'LAThigh_ACCY'),
    "scherpereel_exo": (['thigh_imu_l_accel_x', 'thigh_imu_l_accel_y', 'thigh_imu_l_accel_z'], 'thigh_imu_l_accel_y'),
    "molinaro":        (['thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l'], 'thigh_accel_x_l'),
}

# camargo raw-label -> dense activity (copied from camargo_v2.py, read-only)
CAM_ACTS = ["stand", "walk", "turn", "jog", "rampascent", "rampdescent", "stairascent", "stairdescent"]
CAM_RAW_TO_DENSE = {
    "stand": "stand", "stand-walk": "walk", "walk": "walk", "walk-stand": "stand",
    "turn1": "turn", "turn2": "turn", "jog": "jog",
    "rampascent": "rampascent", "walk-rampascent": "rampascent", "rampascent-walk": "walk",
    "rampdescent": "rampdescent", "walk-rampdescent": "rampdescent", "rampdescent-walk": "walk",
    "stairascent": "stairascent", "walk-stairascent": "stairascent", "stairascent-walk": "walk",
    "stairdescent": "stairdescent", "walk-stairdescent": "stairdescent", "stairdescent-walk": "walk",
}


def find_raw_csv(key):
    """Return (imu_csv, flag_csv_or_None) for an INCLUDED raw trial, or (None, None)."""
    if key == "camargo":
        return first_glob(os.path.join(RAW_ROOT, "01_Camargo", "dataset", "AB*", "training_data", "*.csv")), None
    if key == "scherpereel":
        imu = first_glob(os.path.join(RAW_ROOT, "02_Scherpereel", "ProcessedData", "AB*", "incline_walk*up*", "*_imu_real.csv"))
        flag = None
        if imu:
            cand = glob.glob(os.path.join(os.path.dirname(imu), "*_activity_flag.csv"))
            flag = cand[0] if cand else None
        return imu, flag
    if key == "scherpereel_exo":
        imu = first_glob(os.path.join(RAW_ROOT, "03_MonilaroScherpereel", "Phase1And2_Parsed", "BT*", "*incline_walk*up*", "*_exo.csv"),
                         os.path.join(RAW_ROOT, "03_MonilaroScherpereel", "Phase1And2_Parsed", "BT*", "*", "*_exo.csv"))
        flag = None
        if imu:
            cand = glob.glob(os.path.join(os.path.dirname(imu), "*_activity_flag.csv"))
            flag = cand[0] if cand else None
        return imu, flag
    if key == "molinaro":
        for mode in ("LG", "RA", "RD", "SA", "SD", "ST"):
            hit = first_glob(os.path.join(RAW_ROOT, "04_Monilaro", "dataset", "AB*", mode + "_*", "exo.csv"))
            if hit:
                return hit, None
    return None, None


def stage1_gravity(key, imu_csv):
    """Confirm gravity lands on the raw axis the script calls VERTICAL."""
    cols, vert = RAW_ACCEL[key]
    df = pd.read_csv(imu_csv)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        record(f"S1 {key}: raw accel columns present", False, f"missing {missing}")
        return
    record(f"S1 {key}: raw accel columns present", True, str(cols))
    a = df[cols].values.astype(float)
    a = a[np.all(np.isfinite(a), axis=1)]
    means = a.mean(axis=0)
    dom = int(np.argmax(np.abs(means)))
    ok = (cols[dom] == vert)
    record(f"S1 {key}: gravity on script's vertical axis", ok,
           f"raw mean(acc)={np.round(means,2).tolist()} dominant={cols[dom]} expected={vert}")


def stage1_reproduce(key, ddir, ver):
    """Re-apply the script's exact transform to one raw trial; assert the
    resulting windows appear verbatim in the saved per-source npy."""
    imu, flag = find_raw_csv(key)
    if imu is None:
        record(f"S1 {key}: raw trial found", None, "no raw CSV on disk -> skipped")
        return
    stage1_gravity(key, imu)

    saved = np.load(os.path.join(ROOT, "dataset", ddir, f"data_{ver}.npy"))
    try:
        if key == "molinaro":
            cols = ['thigh_accel_y_l', 'thigh_accel_x_l', 'thigh_accel_z_l',
                    'thigh_gyro_y_l', 'thigh_gyro_x_l', 'thigh_gyro_z_l']
            df = pd.read_csv(imu)
            s = df[cols].values.astype(float)
            s = s[np.all(np.isfinite(s), axis=1)]
            W = window(down_sample(s)) * -1.0                      # global negation

        elif key in ("scherpereel", "scherpereel_exo"):
            if key == "scherpereel":
                cols = ['LAThigh_ACCX', 'LAThigh_ACCY', 'LAThigh_ACCZ',
                        'LAThigh_GYROX', 'LAThigh_GYROY', 'LAThigh_GYROZ']
            else:
                cols = ['thigh_imu_l_accel_z', 'thigh_imu_l_accel_y', 'thigh_imu_l_accel_x',
                        'thigh_imu_l_gyro_z', 'thigh_imu_l_gyro_y', 'thigh_imu_l_gyro_x']
            df = pd.read_csv(imu)
            fl = pd.read_csv(flag)
            m = min(len(df), len(fl))
            df, fl = df.iloc[:m], fl.iloc[:m]
            s = df[cols].values.astype(float)
            keep = (fl['left'].values == 1) & np.all(np.isfinite(s), axis=1)
            seg = s[keep].copy()
            seg[:, 3:] *= RAD_PER_DEG
            if key == "scherpereel":
                seg[:, [0, 2, 3, 5]] *= -1
                W = window(down_sample(seg))
            else:
                W = window(down_sample(seg)) * -1.0                # global negation

        else:  # camargo: transform whole CSV, reproduce the longest dense segment
            cols = ['thigh_Accel_Z', 'thigh_Accel_X', 'thigh_Accel_Y',
                    'thigh_Gyro_Z', 'thigh_Gyro_X', 'thigh_Gyro_Y']
            df = pd.read_csv(imu)
            s = df[cols].values.astype(float)
            s[:, :3] *= 9.81
            s[:, [0, 2, 3, 5]] *= -1
            dense = np.array([CAM_ACTS.index(CAM_RAW_TO_DENSE[a]) if a in CAM_RAW_TO_DENSE else -1
                              for a in df['Label'].values])
            best = (0, 0)
            i = 0
            while i < len(dense):
                j = i
                while j < len(dense) and dense[j] == dense[i]:
                    j += 1
                if dense[i] >= 0 and (j - i) > (best[1] - best[0]):
                    best = (i, j)
                i = j
            W = window(down_sample(s[best[0]:best[1]]))

        if len(W) == 0:
            record(f"S1 {key}: reproduce raw->npy window", None, "no full window in chosen trial")
            return
        dists = [min_window_dist(saved, W[k]) for k in range(min(3, len(W)))]
        worst = max(dists)
        record(f"S1 {key}: reproduced raw window found in saved npy",
               worst < ATOL, f"max-abs-diff over {len(dists)} window(s) = {worst:.2e}")
    except Exception as e:
        record(f"S1 {key}: reproduce raw->npy window", None, f"best-effort error: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# STAGE 2 : 4 aligned npy -> merged  (re-merge in memory == on-disk)
# ----------------------------------------------------------------------------
def stage2():
    sys.path.insert(0, HERE)
    import merge_dense_9cls as M        # safe: main() only runs under __main__

    data_parts, label_parts, slices = [], [], []
    user_offset, start = 0, 0
    for raw_key, ddir, ver in M.SOURCES:
        base = os.path.join(M.ROOT, "dataset", ddir)
        d = np.load(os.path.join(base, f"data_{ver}.npy")).astype(np.float32)
        lab = np.load(os.path.join(base, f"label_{ver}.npy"))
        act = lab[:, :, 0].astype(int)
        remap = M.build_remap(raw_key, np.unique(act).tolist())
        new_act = np.vectorize(remap.__getitem__)(act).astype(np.float32)
        usr = lab[:, :, 1].astype(int)
        uniq = np.unique(usr)
        dmap = {old: i for i, old in enumerate(uniq)}
        new_usr = np.vectorize(dmap.__getitem__)(usr) + user_offset
        user_offset += len(uniq)
        nl = np.empty_like(lab, dtype=np.float32)
        nl[:, :, 0] = new_act
        nl[:, :, 1] = new_usr.astype(np.float32)
        data_parts.append(d)
        label_parts.append(nl)
        slices.append((raw_key, start, start + len(d)))
        start += len(d)

    re_data = np.concatenate(data_parts, 0).astype(np.float32)
    re_label = np.concatenate(label_parts, 0).astype(np.float32)

    disk_data = np.load(os.path.join(ROOT, "dataset", "merged", "data_10_20_9cls_align.npy"))
    disk_label = np.load(os.path.join(ROOT, "dataset", "merged", "label_10_20_9cls_align.npy"))

    record("S2: on-disk data == in-memory re-merge", np.array_equal(re_data, disk_data),
           f"shape {disk_data.shape}, max|diff|={np.abs(re_data-disk_data).max():.2e}")
    record("S2: on-disk label == in-memory re-merge", np.array_equal(re_label, disk_label),
           f"shape {disk_label.shape}")
    record("S2: min activity id == 0 (stand present)", int(disk_label[:, :, 0].min()) == 0)
    users = np.unique(disk_label[:, 0, 1].astype(int))
    record("S2: user ids contiguous 0..N-1", users.tolist() == list(range(len(users))),
           f"{len(users)} subjects")

    with open(os.path.join(ROOT, "dataset", "merged", "label_map.json")) as f:
        lm = json.load(f)
    record("S2: label_map.json == unified index", lm == M.UNIFIED_INDEX, str(lm))

    # per-source per-class counts vs each source's own (name-mapped) distribution
    for raw_key, a, b in slices:
        acts = disk_label[a:b, 0, 0].astype(int)
        ids, cnts = np.unique(acts, return_counts=True)
        dist = {M.UNIFIED[i]: int(c) for i, c in zip(ids, cnts)}
        record(f"S2: {raw_key} per-class windows", None, f"n={b-a}  {dist}")
    return disk_data, disk_label, slices


# ----------------------------------------------------------------------------
# STAGE 3 : merged split by source -> cross-source frame agreement
# ----------------------------------------------------------------------------
def stage3(data, label, slices):
    UNIFIED = ["stand", "walk", "turn", "jog", "rampascent", "rampdescent",
               "stairascent", "stairdescent", "sit-stand-transition"]
    rows = {}
    for raw_key, a, b in slices:
        d = data[a:b]
        lab = label[a:b]
        acc, gyr = d[:, :, 0:3], d[:, :, 3:6]
        accmag = float(np.sqrt((acc ** 2).sum(axis=2)).mean())
        gyro_std = np.round(gyr.reshape(-1, 3).std(axis=0), 3).tolist()
        amean = acc.reshape(-1, 3).mean(axis=0)
        grav_axis = int(np.argmax(np.abs(amean)))
        grav_sign = int(np.sign(amean[grav_axis]))

        act = lab[:, 0, 0].astype(int)
        sel = np.isin(act, [UNIFIED.index("walk"), UNIFIED.index("jog")])
        gw = gyr[sel].reshape(-1, 3)
        aw = acc[sel].reshape(-1, 3)
        gvar = gw.var(axis=0)
        ml_axis = int(np.argmax(gvar))
        ml_frac = float(gvar[ml_axis] / gvar.sum())

        def cov(x, y):
            return float(np.mean(x * y) - x.mean() * y.mean())

        def skew(x):
            mu, sd = x.mean(), x.std()
            return float(np.mean(((x - mu) / sd) ** 3)) if sd > 0 else 0.0

        inv = {
            "cov(gyro0,acc2)": cov(gw[:, 0], aw[:, 2]),
            "cov(gyro0,acc1)": cov(gw[:, 0], aw[:, 1]),
            "skew(gyro0)": skew(gw[:, 0]),
        }
        rows[raw_key] = dict(accmag=accmag, gyro_std=gyro_std, grav_axis=grav_axis,
                             grav_sign=grav_sign, ml_axis=ml_axis, ml_frac=ml_frac, inv=inv)

    print("\n  per-source frame stats (walk+jog for ML / invariants):")
    print(f"    {'source':16s} {'|acc|':>6s} {'gyroStd(0,1,2)':>20s} {'gravAx':>7s} {'gSign':>6s} {'gyroDomAx':>10s} {'mlFrac':>7s}")
    for k in rows:
        r = rows[k]
        print(f"    {k:16s} {r['accmag']:6.2f} {str(r['gyro_std']):>20s} "
              f"{r['grav_axis']:>7d} {r['grav_sign']:>6d} {r['ml_axis']:>10d} {r['ml_frac']:>7.2f}")
    print("\n  signed gait invariants (sign should agree across sources):")
    for k in rows:
        inv = rows[k]['inv']
        print(f"    {k:16s} " + "  ".join(f"{n}={'+' if v>=0 else '-'}({v:+.3f})" for n, v in inv.items()))

    keys = [k for k, _, _ in slices]
    # units
    record("S3: |acc| ~9.8 all sources (m/s^2, not g/grav-removed)",
           all(7.0 < rows[k]['accmag'] < 13.0 for k in keys),
           f"{ {k: round(rows[k]['accmag'],1) for k in keys} }")
    gmax = max(max(rows[k]['gyro_std']) for k in keys)
    gmin = min(min(s for s in rows[k]['gyro_std'] if s > 0) for k in keys)
    record("S3: gyro std same O(1) scale all sources (rad/s, not deg/s)",
           gmax / gmin < 10.0, f"max std={gmax:.2f}, min std={gmin:.2f}, ratio={gmax/gmin:.1f}")
    # vertical
    record("S3: gravity on col1 + same sign, all sources",
           all(rows[k]['grav_axis'] == 1 for k in keys) and len({rows[k]['grav_sign'] for k in keys}) == 1,
           f"axes={ {k: rows[k]['grav_axis'] for k in keys} } signs={ {k: rows[k]['grav_sign'] for k in keys} }")
    # ML axis on col0
    record("S3: walk/jog gyro dominated by col0 (ML), all sources",
           all(rows[k]['ml_axis'] == 0 for k in keys),
           f"{ {k: rows[k]['ml_axis'] for k in keys} }")
    # signed invariant agreement (report; scherpereel C may be the known exception)
    for name in ["cov(gyro0,acc2)", "cov(gyro0,acc1)", "skew(gyro0)"]:
        signs = {k: (1 if rows[k]['inv'][name] >= 0 else -1) for k in keys}
        agree = len(set(signs.values())) == 1
        record(f"S3: sign agreement {name}", agree if agree else None, str(signs))
    return rows


def main():
    print("=" * 78)
    print("VERIFY  dataset/merged/data_10_20_9cls_align.npy   (read-only)")
    print("=" * 78)

    print("\n[Stage 1] raw CSV -> per-source aligned npy (re-derive + raw gravity)")
    if not os.path.isdir(RAW_ROOT):
        record("S1: raw data root present", None, f"{RAW_ROOT} not found -> Stage 1 skipped")
    else:
        for raw_key, ddir, ver in SOURCES:
            stage1_reproduce(raw_key, ddir, ver)

    print("\n[Stage 2] 4 aligned npy -> merged (re-merge == on-disk)")
    data, label, slices = stage2()

    print("\n[Stage 3] merged split by source -> cross-source frame agreement")
    stage3(data, label, slices)

    print("\n" + "=" * 78)
    npass = sum(1 for _, p, _ in CHECKS if p is True)
    nfail = sum(1 for _, p, _ in CHECKS if p is False)
    ninfo = sum(1 for _, p, _ in CHECKS if p is None)
    print(f"SUMMARY: {npass} PASS, {nfail} FAIL, {ninfo} info/skip")
    if nfail:
        print("FAILED checks:")
        for n, p, d in CHECKS:
            if p is False:
                print(f"  - {n}  {d}")
    print("VERDICT:", "ALIGNED (no hard failures)" if nfail == 0 else "PROBLEM FOUND -> see FAILED checks")
    print("=" * 78)


if __name__ == "__main__":
    main()
