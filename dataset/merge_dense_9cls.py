"""
Merge 4 leg/thigh-IMU datasets into ONE dense 9-class dataset (offline).

Unlike the on-the-fly `--merge` used in ssl_compare/pretrain_ssl.py (which just
concatenates pre-split arrays for SSL pretraining), this script builds a single
PERSISTED dataset on disk with a unified, name-aligned 9-class activity vocabulary
so it can be trained/evaluated like any other entry in data_config.json.

Sources (all .npy already generated; each label is (N, seq_len, 2)=[activity, user]):
    camargo_10_20_dense_8cls               (8 classes)
    scherpereel_10_20_both_dense_9cls      (9 classes)
    scherpereel_exo_10_20_both_dense_9cls  (9 classes)
    molinaro_10_20_both_dense_7cls         (7 classes)

CRITICAL: the per-source raw npy activity ids are remapped by CLASS NAME using each
dataset's OWN label map (RAW_ID2NAME below) -- NOT data_config.json. The config's
`activity_label` order was deliberately permuted (e.g. scherpereel's "stand" parked
last) to line up with the `- np.min` shift that utils.partition_* applies during
single-dataset training; scherpereel/exo have NO stand samples and their npy ids
start at 1 (walk=1, ... , sit-stand-transition=8). So config order does NOT reflect
raw ids and must not be used for the merge. Source of truth:
    camargo  -> dataset/camargo_v2.py    ACTIVITY_NAMES (label = list index)
    scherp.  -> dataset/scherpereel/label_map.json
    exo      -> dataset/scherpereel_exo.py DENSE_ACTIVITIES (build_label_map = {name:i})
    molinaro -> dataset/molinaro/label_map.json

Target unified vocabulary (id == list position):
    0 stand   1 walk   2 turn   3 jog   4 rampascent
    5 rampdescent   6 stairascent   7 stairdescent   8 sit-stand-transition
'stand' is kept at id 0 so the merged set's min activity id is 0 (camargo/molinaro
provide stand), avoiding the `- np.min` shift in utils.partition_* on the merged set.

User labels collide across datasets (each starts near 0), so they are densified to
0..k-1 per source and shifted by a cumulative offset to make every subject globally
unique. Label shape stays (N, seq_len, 2): index 0 = unified activity, index 1 =
global user id, so the merged set can still be grouped-CV split by subject
(utils.partition_grouped_and_reshape, group_label_index=1) without cross-dataset
collisions.

Usage:
    python dataset/merge_dense_9cls.py
"""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Target unified vocabulary (id == list position).
UNIFIED = [
    "stand", "walk", "turn", "jog", "rampascent",
    "rampdescent", "stairascent", "stairdescent", "sit-stand-transition",
]
UNIFIED_INDEX = {name: i for i, name in enumerate(UNIFIED)}

# Authoritative raw npy activity-id -> class name, per source (see module docstring).
RAW_ID2NAME = {
    "camargo": {
        0: "stand", 1: "walk", 2: "turn", 3: "jog",
        4: "rampascent", 5: "rampdescent", 6: "stairascent", 7: "stairdescent",
    },
    "scherpereel": {
        1: "walk", 2: "turn", 3: "jog", 4: "rampascent",
        5: "rampdescent", 6: "stairascent", 7: "stairdescent", 8: "sit-stand-transition",
    },
    "scherpereel_exo": {
        1: "walk", 2: "turn", 3: "jog", 4: "rampascent",
        5: "rampdescent", 6: "stairascent", 7: "stairdescent", 8: "sit-stand-transition",
    },
    "molinaro": {
        0: "walk", 1: "jog", 2: "rampascent", 3: "rampdescent",
        4: "stairascent", 5: "stairdescent", 6: "stand",
    },
}

# (raw_map_key, dataset_dir, file_version)  ->  file = dataset/<dir>/data_<version>.npy
SOURCES = [
    ("camargo",         "camargo",         "10_20_dense_8cls_zxy"),
    ("scherpereel",     "scherpereel",     "10_20_both_dense_9cls_-xy-z"),
    ("scherpereel_exo", "scherpereel_exo", "10_20_both_dense_9cls_-z-y-x"),
    ("molinaro",        "molinaro",        "10_20_both_dense_7cls_-y-x-z"),
]

OUT_DIR = os.path.join(ROOT, "dataset", "merged")
OUT_VERSION = "10_20_9cls_align"          # -> data_10_20_merged_9cls.npy / label_...
OUT_KEY = "merged_10_20_9cls_align"       # data_config.json key (dataset="merged" + version)


def build_remap(raw_key, present_ids):
    """raw npy id -> unified id, by class name. Errors if a present id is unmapped."""
    id2name = RAW_ID2NAME[raw_key]
    missing = sorted(set(present_ids) - set(id2name))
    if missing:
        raise ValueError(f"{raw_key}: npy has activity ids {missing} with no name in RAW_ID2NAME")
    return {raw_id: UNIFIED_INDEX[name] for raw_id, name in id2name.items()}


def main():
    data_parts, label_parts = [], []
    user_offset = 0   # cumulative offset -> globally unique subject ids
    print(f"Unified vocab ({len(UNIFIED)} classes): {UNIFIED_INDEX}\n")

    for raw_key, ddir, ver in SOURCES:
        base = os.path.join(ROOT, "dataset", ddir)
        data = np.load(os.path.join(base, f"data_{ver}.npy")).astype(np.float32)
        label = np.load(os.path.join(base, f"label_{ver}.npy"))

        # Remap activity ids (index 0) by name, using each source's OWN raw map.
        act = label[:, :, 0].astype(int)
        remap = build_remap(raw_key, np.unique(act).tolist())
        new_act = np.vectorize(remap.__getitem__)(act).astype(np.float32)

        # Densify user ids to 0..k-1 then shift by cumulative offset -> globally unique.
        usr = label[:, :, 1].astype(int)
        uniq = np.unique(usr)
        dense_map = {old: i for i, old in enumerate(uniq)}
        new_usr = np.vectorize(dense_map.__getitem__)(usr) + user_offset
        user_offset += len(uniq)

        new_label = np.empty_like(label, dtype=np.float32)
        new_label[:, :, 0] = new_act
        new_label[:, :, 1] = new_usr.astype(np.float32)

        data_parts.append(data)
        label_parts.append(new_label)

        ids, counts = np.unique(new_act[:, 0].astype(int), return_counts=True)
        dist = {UNIFIED[i]: int(c) for i, c in zip(ids, counts)}
        print(f"{raw_key:18s} n={data.shape[0]:6d}  users={len(uniq):2d}  {dist}")

    data = np.concatenate(data_parts, 0).astype(np.float32)
    label = np.concatenate(label_parts, 0).astype(np.float32)

    ids, counts = np.unique(label[:, 0, 0].astype(int), return_counts=True)
    merged_dist = {UNIFIED[i]: int(c) for i, c in zip(ids, counts)}
    print(f"\nMerged: data={data.shape}  label={label.shape}")
    print(f"Per-class windows: {merged_dist}")
    assert int(label[:, :, 0].min()) == 0, "min activity id must be 0 (stand present)"

    # user ids must be a contiguous 0..user_offset-1 block (globally unique subjects).
    merged_users = np.unique(label[:, 0, 1].astype(int))
    assert merged_users.tolist() == list(range(user_offset)), "user ids not contiguous/unique"
    print(f"Global subjects: {user_offset} (ids 0..{user_offset - 1})")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"data_{OUT_VERSION}.npy"), data)
    np.save(os.path.join(OUT_DIR, f"label_{OUT_VERSION}.npy"), label)
    with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
        json.dump(UNIFIED_INDEX, f, indent=2)

    entry = {OUT_KEY: {
        "sr": 10, "seq_len": 20, "dimension": 6,
        "activity_label_index": 0, "activity_label_size": len(UNIFIED),
        "activity_label": list(UNIFIED),
        "user_label_index": 1, "user_label_size": int(user_offset),
        "size": int(data.shape[0]),
    }}
    print("\nAdd to data_config.json:")
    print(json.dumps(entry, indent=4))


if __name__ == "__main__":
    main()
