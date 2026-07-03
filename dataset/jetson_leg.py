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
* 7 dense activity classes (no rampdescent):
  ["stand", "walk", "turn", "jog", "rampascent", "stairascent", "stairdescent"]
  -> compact ids 0..6. label.csv tokens are normalised via LABEL_TO_DENSE.
* Position (leg- vs pocket-mounted) is the parent folder, selected with
  --position; one NPY is built per position (mount distributions kept separate).
* Subject (AB0x) is written into the label as user_id (0-based over the subjects
  discovered for that position), so user_label_size = number of subjects.
* Units already match the camargo/molinaro NPYs (accel m/s^2, gyro rad/s),
  so NO unit conversion.
* Block-averaging downsample ~63 Hz -> 10 Hz (no anti-alias filter, matching
  how the foundation model was pretrained). Each contiguous activity segment is
  downsampled and windowed independently so no window straddles a label change.
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

# label.csv token -> dense activity name. make_labels.py writes already-dense
# tokens (stand/walk/turn/jog/rampascent/stairascent/stairdescent); "rampup" is
# kept for backward compatibility with older hand-written labels.
LABEL_TO_DENSE = {
    "stand": "stand", "walk": "walk", "turn": "turn", "jog": "jog",
    "rampascent": "rampascent", "rampup": "rampascent",
    "stairascent": "stairascent", "stairdescent": "stairdescent",
}

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


def preprocess(path, path_save, version, leg, position, subjects_filter=None,
               raw_sr=RAW_SR, target_sr=TARGET_SR, seq_len=SEQ_LEN):
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
    for subj, folder, accel_path, gyro_path, label_path in trials:
        dense, sensors = load_trial_sides(accel_path, gyro_path, label_path, sides)
        n_win = 0
        for side in sides:
            data, acts = segment_and_window(dense, sensors[side],
                                            seq_len, raw_sr, target_sr)
            for d, a in zip(data, acts):
                lbl = np.zeros((d.shape[0], seq_len, 2))
                lbl[:, :, 0] = a[:, None]
                lbl[:, :, 1] = subj_to_id[subj]
                data_all.append(d)
                label_all.append(lbl)
                n_win += d.shape[0]
        print(f'  {subj}/{os.path.basename(folder):28s} -> {n_win} windows')

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
    p.add_argument('--subjects', nargs='+', default='AB02',
                   help='One or more subjects to include, e.g. --subjects AB02 '
                        'AB03 (also accepts 2 / 02 / ab02). Omit for all subjects.')
    p.add_argument('--tgt_sr', type=int, default=TARGET_SR)
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    args = p.parse_args()

    suffix  = '' if args.leg == 'left' else f'_{args.leg}'
    if args.subjects:
        subj_tag = ''.join(normalize_subject(s)[2:] for s in sorted(args.subjects))
        suffix += f'_{subj_tag}'
    version = f'{args.tgt_sr}_{args.seq_len}{suffix}_xyz_{args.position}_AB02'

    preprocess(args.input_dir, 'dataset/jetson_leg', version, args.leg, args.position,
               subjects_filter=args.subjects,
               target_sr=args.tgt_sr, seq_len=args.seq_len)
