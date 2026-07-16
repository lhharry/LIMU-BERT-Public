"""
Jetson bilateral IMU recordings -> LIMU-BERT NPY format.

Source
------
DATA/jetson/AB0x/{Leg,Pocket}/<HH_MM_SS>_<token>/  (subject / position / trial).
Each trial folder has:
    accelerometers.csv  Time, Left_{x,y,z}, Right_{x,y,z}   (m/s^2)
    gyroscopes.csv      Time, Left_{x,y,z}, Right_{x,y,z}   (rad/s)
    label.csv           Time, Label                         (per-sample)
label.csv is produced by DATA/jetson/make_labels.py (gyro-energy stand
detection) and has a stand -> activity -> stand structure with timestamps
identical to the sensors. Run make_labels.py first for any subject that ships
without labels (e.g. AB02); this loader errors if a trial has no label.csv.

Conventions (aligned with molinaro.py / camargo_v2.py / jetson_compare.py)
-------------------------------------------------------------------------
* 11 dense activity classes:
  ["stand", "walk", "turn", "jog", "rampascent", "stairascent", "stairdescent",
   "sit", "sit-to-stand", "stand-to-sit", "rampdescent"]
  -> compact ids 0..10. The original 7 classes keep their ids 0..6; the four
  new classes (sit / sit-to-stand / stand-to-sit / rampdescent, added with the
  SiSt/StSi/rampdown recordings) are appended at the end so NPYs built before
  the extension stay id-compatible. label.csv tokens are normalised via
  LABEL_TO_DENSE.
* Rows holding any non-finite sensor value (NaN, e.g. from empty CSV cells)
  are skipped: the row is dropped from every selected side so windowing never
  sees it, and the total dropped rows are reported per class at the end.
* Transition classes (sit-to-stand / stand-to-sit) last ~1-1.5 s, shorter than
  one 2 s window, so per-segment windowing would emit none. Each transition
  segment instead yields exactly ONE window centered on it: longer segments
  are cut to the middle seq_len frames, shorter ones are padded with real
  neighbouring sit/stand rows, and the window is kept only if >= MIN_TRANS_FRAC
  (50%) of its frames are transition frames -- otherwise the segment is
  dropped and reported. sit/stand windows themselves stay pure per-segment
  windows and never contain transition rows.
* Position (leg- vs pocket-mounted) is the parent folder, selected with
  --position; one NPY is built per position (mount distributions kept separate).
* Subject (AB0x) is written into the label as user_id (0-based over the subjects
  discovered for that position), so user_label_size = number of subjects.
* Units already match the camargo/molinaro NPYs (accel m/s^2, gyro rad/s),
  so NO unit conversion.
* Block-averaging downsample raw -> 10 Hz (no anti-alias filter, matching how
  the foundation model was pretrained). The raw rate is auto-detected per trial
  from the Time column (trials range ~62-100 Hz; AB01 mixes rates across trials),
  so every trial truly lands at 10 Hz / 2 s windows; pass --raw_sr to override.
  Each contiguous activity segment is downsampled and windowed independently so
  no window straddles a label change.
* --leg left|right|both: each chosen leg becomes its own 6-dim sample stream
  (both -> Left and Right pooled as separate samples; version suffix _both).
* label shape (N, seq_len, 2) = [activity_id, user_id].

Usage (run from the LIMU-BERT-Public repo root):
    python dataset/jetson_leg.py --leg both --position leg
    python dataset/jetson_leg.py --leg both --position pocket
    python dataset/jetson_leg.py --leg both --position leg --subjects AB02 AB03
By default every AB* subject under the position is included; --subjects restricts
to the listed subjects (a subject tag is appended to the version so subset NPYs
don't overwrite the full one).
"""

import os
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

# The jetson trials do NOT share one sampling rate: older trials run ~62-66 Hz
# while some newer AB01 trials run ~100 Hz, and AB01 even mixes both rates across
# its own trials. So raw_sr is auto-detected per trial from the Time column (see
# estimate_sr) instead of being a global constant. RAW_SR below is only a
# fallback used when detection fails and no --raw_sr override is given.
RAW_SR    = 100    # fallback sampling rate (Hz) when per-trial detection fails
TARGET_SR = 10
SEQ_LEN   = 20

DATASET_PATH = r'D:\01_Code\DATA\jetson'

# Dense 11-class label space (compact ids 0..10). The first 7 ids match the
# pre-extension NPYs; the new classes are appended so existing ids never move.
ACTIVITY_NAMES = ["stand", "walk", "turn", "jog",
                  "rampascent", "stairascent", "stairdescent",
                  "sit", "sit-to-stand", "stand-to-sit", "rampdescent"]

# label.csv token -> dense activity name. make_labels.py writes already-dense
# tokens; "rampup"/"rampdown" are kept for backward compatibility with older
# hand-written labels.
LABEL_TO_DENSE = {
    "stand": "stand", "walk": "walk", "turn": "turn", "jog": "jog",
    "rampascent": "rampascent", "rampup": "rampascent",
    "stairascent": "stairascent", "stairdescent": "stairdescent",
    "sit": "sit", "sit-to-stand": "sit-to-stand",
    "stand-to-sit": "stand-to-sit",
    "rampdescent": "rampdescent", "rampdown": "rampdescent",
}

# The transition classes are shorter (~1-1.5 s) than one 2 s window, so pure
# per-segment windowing would never emit them. Each transition segment instead
# yields exactly one window centered on it, padded with real neighbouring
# sit/stand rows (or cut to the middle if longer than a window); the window is
# kept only when at least MIN_TRANS_FRAC of its frames are transition frames.
TRANSITION_NAMES = {"sit-to-stand", "stand-to-sit"}
MIN_TRANS_FRAC   = 0.5

# Per-side source columns in the jetson-native (x, y, z) order, accel first then
# gyro. NOTE: this is NOT the camargo remap (y, x, z) that
# jetson_compare.load_jetson_trial applies; gravity sits on the raw y axis here.
def side_cols(side):
    return [f"{side}_x", f"{side}_y", f"{side}_z"]

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


def estimate_sr(time_strings):
    """
    Estimate a trial's sampling rate (Hz) from its Time column, which holds wall
    clock 'HH:MM:SS.ffffff' strings. Returns 1 / median(positive inter-sample dt),
    or None if there are fewer than 3 samples or no positive gap (caller falls
    back to --raw_sr / RAW_SR). Trials are short and within one hour, so midnight
    wrap is not handled; non-positive diffs are dropped.
    """
    secs = []
    for s in time_strings:
        t = datetime.strptime(str(s).strip(), "%H:%M:%S.%f")
        secs.append(t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6)
    if len(secs) < 3:
        return None
    dt = np.diff(secs)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return 1.0 / float(np.median(dt))


def normalize_subject(token):
    """'2' / '02' / 'ab02' / 'AB02' -> 'AB02'. Non-numeric tokens are upper-cased."""
    t = token.strip()
    if t.lower().startswith('ab'):
        t = t[2:]
    return f'AB{int(t):02d}' if t.isdigit() else token.strip().upper()


def discover_trials(root, position, subjects=None):
    """
    Return sorted [(subject, folder, accel_path, gyro_path, label_path)] for every
    trial under root/AB*/<Position>/, where <Position> matches `position`
    case-insensitively. If `subjects` is given (list of normalised AB tokens),
    only those subjects are included. Errors if a trial lacks label.csv
    (run make_labels.py).
    """
    want = None if not subjects else {normalize_subject(s) for s in subjects}
    out = []
    for subject in sorted(glob.glob(os.path.join(root, 'AB*'))):
        if not os.path.isdir(subject):
            continue
        subj = os.path.basename(subject)
        if want is not None and normalize_subject(subj) not in want:
            continue
        for pos_dir in glob.glob(os.path.join(subject, '*')):
            if not (os.path.isdir(pos_dir)
                    and os.path.basename(pos_dir).lower() == position.lower()):
                continue
            for dirpath in sorted(glob.glob(os.path.join(pos_dir, '*'))):
                if not os.path.isdir(dirpath):
                    continue
                accel = glob.glob(os.path.join(dirpath, 'accelerometers*.csv'))
                gyro  = os.path.join(dirpath, 'gyroscopes.csv')
                if not accel or not os.path.isfile(gyro):
                    continue
                label = os.path.join(dirpath, 'label.csv')
                if not os.path.isfile(label):
                    raise SystemExit(
                        f'No label.csv in {dirpath}; run DATA/jetson/make_labels.py first.')
                out.append((subj, dirpath, accel[0], gyro, label))
    return out


def load_trial_sides(accel_path, gyro_path, label_path, sides):
    """
    Return (dense_labels[list], {side: (N,6) float array}, raw_sr, skipped) for
    one trial, truncated to the common length of the three files. raw_sr is
    auto-detected from the (truncated) accel Time column, or None if it cannot
    be estimated. Rows with any non-finite sensor value in any selected side
    are dropped from dense and every side alike (so one CSV row counts once no
    matter how many sides are loaded); `skipped` maps dense activity name ->
    number of rows dropped that way.
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

    # Estimate sr before NaN-row filtering: the Time column of dropped rows is
    # still valid, and the median makes the estimate robust anyway.
    raw_sr = estimate_sr(acc['Time']) if 'Time' in acc.columns else None

    sensors = {}
    for side in sides:
        cols = side_cols(side)
        a = acc[cols].to_numpy(dtype=float)
        g = gyr[cols].to_numpy(dtype=float)
        sensors[side] = np.hstack([a, g])     # (N,6): accel xyz, gyro xyz

    keep = np.ones(n, dtype=bool)
    for mat in sensors.values():
        keep &= np.isfinite(mat).all(axis=1)
    skipped = Counter()
    if not keep.all():
        for d, k in zip(dense, keep):
            if not k:
                skipped[d] += 1
        dense = [d for d, k in zip(dense, keep) if k]
        sensors = {side: mat[keep] for side, mat in sensors.items()}
    return dense, sensors, raw_sr, skipped


def segment_and_window(dense, sensor, seq_len, raw_sr, target_sr):
    """
    Split one (N,6) stream into fixed-length windows per contiguous activity
    segment; transition segments (TRANSITION_NAMES) instead yield one centered
    window each (see the constant's comment). Returns (data_list,
    label_act_list, trans_kept, trans_dropped); the trans_* Counters record,
    per transition class, how many segments produced a window vs. were dropped
    (below MIN_TRANS_FRAC or trial too short). User id is added by caller.
    """
    data, acts = [], []
    trans_kept, trans_dropped = Counter(), Counter()
    n = len(dense)
    r = raw_sr * 1.0 / target_sr
    i = 0
    while i < n:
        j = i
        while j < n and dense[j] == dense[i]:
            j += 1
        name = dense[i]
        # Non-finite rows were already dropped in load_trial_sides.
        if name in TRANSITION_NAMES:
            # One window centered on the segment; ask for one spare block so
            # down_sample's non-integer remainder handling never leaves us a
            # frame short, then crop to the middle seq_len frames.
            need_raw = int(math.ceil(seq_len * r + r))
            lo = int(round((i + j - need_raw) / 2.0))
            lo = min(max(lo, 0), n - need_raw)
            if lo < 0:
                trans_dropped[name] += 1
                i = j
                continue
            down = down_sample(sensor[lo:lo + need_raw], raw_sr, target_sr)
            if down.shape[0] < seq_len:
                trans_dropped[name] += 1
                i = j
                continue
            s0 = (down.shape[0] - seq_len) // 2
            # Transition fraction of the cropped window, via its raw extent.
            win_lo = lo + s0 * r
            win_hi = win_lo + seq_len * r
            overlap = min(j, win_hi) - max(i, win_lo)
            if overlap / r < MIN_TRANS_FRAC * seq_len:
                trans_dropped[name] += 1
                i = j
                continue
            data.append(down[s0:s0 + seq_len][None])       # (1, seq_len, 6)
            acts.append(np.full(1, ACTIVITY_NAMES.index(name)))
            trans_kept[name] += 1
        else:
            seg = sensor[i:j]
            down = down_sample(seg, raw_sr, target_sr)
            n_win = down.shape[0] // seq_len
            if n_win:
                down = down[:n_win * seq_len].reshape(n_win, seq_len, 6)
                data.append(down)
                acts.append(np.full(n_win, ACTIVITY_NAMES.index(dense[i])))
        i = j
    return data, acts, trans_kept, trans_dropped


def preprocess(path, path_save, version, leg, position, subjects_filter=None,
               raw_sr=None, target_sr=TARGET_SR, seq_len=SEQ_LEN):
    # raw_sr None -> auto-detect per trial from the Time column; a numeric value
    # forces that rate for every trial (overriding detection).
    # All subject folders present (unfiltered) -> used for helpful error messages.
    available = sorted(os.path.basename(d) for d in glob.glob(os.path.join(path, 'AB*'))
                       if os.path.isdir(d))

    if subjects_filter:
        want = {normalize_subject(s) for s in subjects_filter}
        have = {normalize_subject(s) for s in available}
        missing = sorted(want - have)
        if missing:
            raise SystemExit(
                f'Requested subjects {missing} not found under {path}. '
                f'Available subjects: {available or "(none)"}')

    trials = discover_trials(path, position, subjects_filter)
    if not trials:
        want = '' if not subjects_filter else f' for subjects {subjects_filter}'
        raise SystemExit(
            f'No {position} trials with label.csv found under {path}{want}. '
            f'Available subjects: {available or "(none)"}')
    sides = LEG_SIDES[leg]

    subjects = sorted({subj for subj, *_ in trials})
    subj_to_id = {s: i for i, s in enumerate(subjects)}

    data_all, label_all = [], []
    skipped_total = Counter()
    trans_kept_total, trans_dropped_total = Counter(), Counter()
    for subj, folder, accel_path, gyro_path, label_path in trials:
        dense, sensors, trial_sr, skipped = load_trial_sides(
            accel_path, gyro_path, label_path, sides)
        skipped_total.update(skipped)
        # --raw_sr override wins; else per-trial detection; else RAW_SR fallback.
        if raw_sr is not None:
            sr = raw_sr
        elif trial_sr is not None:
            sr = trial_sr
        else:
            sr = RAW_SR
            print(f'  WARNING: could not detect sr for {folder}; '
                  f'falling back to {RAW_SR} Hz')
        n_win = 0
        for side in sides:
            data, acts, t_kept, t_drop = segment_and_window(
                dense, sensors[side], seq_len, sr, target_sr)
            trans_kept_total.update(t_kept)
            trans_dropped_total.update(t_drop)
            for d, a in zip(data, acts):
                lbl = np.zeros((d.shape[0], seq_len, 2))
                lbl[:, :, 0] = a[:, None]
                lbl[:, :, 1] = subj_to_id[subj]
                data_all.append(d)
                label_all.append(lbl)
                n_win += d.shape[0]
        skip_note = ''
        if skipped:
            counts = ', '.join(f'{k}:{v}' for k, v in sorted(skipped.items()))
            skip_note = f'  (skipped {sum(skipped.values())} NaN rows: {counts})'
        print(f'  {subj}/{os.path.basename(folder):28s} '
              f'sr={sr:6.2f}Hz -> {n_win} windows{skip_note}')

    data  = np.concatenate(data_all, 0).astype(np.float32)
    label = np.concatenate(label_all, 0).astype(np.float32)

    # Window counts per subject x activity.
    print('\nWindows per subject x activity:')
    acts_all = label[:, 0, 0].astype(int)
    users_all = label[:, 0, 1].astype(int)
    for s in subjects:
        uid = subj_to_id[s]
        row = {ACTIVITY_NAMES[i]: int(((acts_all == i) & (users_all == uid)).sum())
               for i in np.unique(acts_all[users_all == uid])}
        print(f'  {s} (id {uid}): {row}')

    # Per-class report of raw rows dropped for non-finite sensor values.
    print('\nNaN rows skipped per class:')
    for name in ACTIVITY_NAMES:
        print(f'  {name}: {skipped_total.get(name, 0)}')
    print(f'  TOTAL: {sum(skipped_total.values())}')

    # Transition segments -> centered windows (counted per side stream).
    print('\nTransition windows (centered %ds, >=%d%% transition frames):'
          % (seq_len // target_sr, int(MIN_TRANS_FRAC * 100)))
    for name in sorted(TRANSITION_NAMES):
        print(f'  {name}: kept {trans_kept_total.get(name, 0)}, '
              f'dropped {trans_dropped_total.get(name, 0)}')

    print('\nAll data processed. Size: %d' % (data.shape[0]))
    print(f'[{leg}/{position}] data={data.shape}  label={label.shape}')
    print(f'subjects: {subj_to_id}')

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
        'user_label_index': 1, 'user_label_size': len(subjects),
        'user_label': subjects,
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
    p.add_argument('--position', choices=['leg', 'pocket'], default='leg')
    p.add_argument('--subjects', nargs='+', default=None,
                   help='One or more subjects to include, e.g. --subjects AB02 '
                        'AB03 (also accepts 2 / 02 / ab02). Omit for all subjects.')
    p.add_argument('--tgt_sr', type=int, default=TARGET_SR)
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    p.add_argument('--raw_sr', type=float, default=None,
                   help='Force this raw sampling rate (Hz) for every trial. '
                        'Omit to auto-detect per trial from the Time column.')
    args = p.parse_args()

    suffix  = '' if args.leg == 'left' else f'_{args.leg}'
    if args.subjects:
        subj_tag = ''.join(normalize_subject(s)[2:] for s in sorted(args.subjects))
        suffix += f'_{subj_tag}'
    version = f'{args.tgt_sr}_{args.seq_len}{suffix}_xyz_{args.position}'

    preprocess(args.input_dir, 'dataset/jetson_leg', version, args.leg, args.position,
               subjects_filter=args.subjects, raw_sr=args.raw_sr,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
