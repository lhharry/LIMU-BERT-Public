#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate + persist the subject-grouped CV fold definitions used by the v3 plan.

The split is fully deterministic from (split_seed, n_folds) via
utils.grouped_fold_assignment, so pretrain (pretrain_ssl.py) and downstream
(bench_eval.py) recompute the SAME folds on the fly -- this script does NOT
create a dependency, it only writes a human-auditable record + a class x subject
coverage report so we can confirm:

  * train/vali/test groups are disjoint and every subject is tested once;
  * every (train) fold contains all classes (no class vanishes group-wise);
  * how imbalanced each fold is per class.

Output: ssl_compare/splits/<dataset>_<version>_grouped_<n>fold_seed<seed>.json

Run from repo root:
  python ssl_compare/make_grouped_splits.py --dataset camargo --dataset_version 10_20_dense_8cls
  python ssl_compare/make_grouped_splits.py --dataset camargo --dataset_version 10_20_dense_8cls --label_index 0
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import grouped_fold_assignment
from config import load_dataset_stats

SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def class_counts(label_act, classes):
    vals, cnts = np.unique(label_act.astype(int), return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, cnts)}
    return {classes[i] if classes and i < len(classes) else str(i): d.get(i, 0)
            for i in range(len(classes) if classes else (int(label_act.max()) + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camargo")
    ap.add_argument("--dataset_version", default="10_20_dense_8cls")
    ap.add_argument("--label_index", type=int, default=0, help="activity label column")
    ap.add_argument("--group_label_index", type=int, default=1, help="subject/user label column")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=3431)
    args = ap.parse_args()

    lpath = os.path.join(REPO_ROOT, "dataset", args.dataset, "label_%s.npy" % args.dataset_version)
    labels = np.load(lpath).astype(np.float32)
    act = labels[:, 0, args.label_index].astype(int)
    grp = labels[:, 0, args.group_label_index].astype(int)

    ds_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    classes = list(ds_cfg.activity_label) if ds_cfg is not None and getattr(ds_cfg, "activity_label", None) else None

    subjects = sorted(np.unique(grp).tolist())
    n_classes = len(classes) if classes else int(act.max()) + 1
    print("dataset=%s_%s  windows=%d  subjects=%d  classes=%d"
          % (args.dataset, args.dataset_version, len(act), len(subjects), n_classes))

    # class x subject coverage matrix
    print("\n#subjects per class (group-splittability check):")
    for c in range(n_classes):
        nsub = len(np.unique(grp[act == c]))
        name = classes[c] if classes else str(c)
        flag = "  <-- <n_folds, cannot fill all folds" if nsub < args.n_folds else ""
        print("  %-14s %6d windows  in %2d subjects%s" % (name, int((act == c).sum()), nsub, flag))

    os.makedirs(SPLIT_DIR, exist_ok=True)
    out = {"dataset": args.dataset, "dataset_version": args.dataset_version,
           "n_folds": args.n_folds, "split_seed": args.split_seed,
           "group_label_index": args.group_label_index, "subjects": subjects, "folds": []}

    seen_test = set()
    for f in range(args.n_folds):
        tr, va, te = grouped_fold_assignment(grp, f, args.n_folds, args.split_seed)
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "fold %d overlap" % f
        seen_test |= te
        tr_mask, va_mask, te_mask = (np.isin(grp, list(s)) for s in (tr, va, te))
        rec = {
            "fold_id": f,
            "train_groups": sorted(tr), "vali_groups": sorted(va), "test_groups": sorted(te),
            "train_windows": int(tr_mask.sum()), "vali_windows": int(va_mask.sum()), "test_windows": int(te_mask.sum()),
            "train_class_counts": class_counts(act[tr_mask], classes),
            "vali_class_counts": class_counts(act[va_mask], classes),
            "test_class_counts": class_counts(act[te_mask], classes),
        }
        out["folds"].append(rec)
        missing = [k for k, v in rec["train_class_counts"].items() if v == 0]
        print("\nfold %d | train=%d vali=%d test=%d subjects | test=%s%s"
              % (f, len(tr), len(va), len(te), sorted(te),
                 ("  !! TRAIN MISSING CLASSES: %s" % missing) if missing else ""))

    assert seen_test == set(subjects), "union of test folds != all subjects"
    print("\nCV sanity: every subject tested exactly once -> OK")

    out_path = os.path.join(SPLIT_DIR, "%s_%s_grouped_%dfold_seed%d.json"
                            % (args.dataset, args.dataset_version, args.n_folds, args.split_seed))
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved -> %s" % out_path)


if __name__ == "__main__":
    main()
