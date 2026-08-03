"""
Carve an unlabeled SSL set out of a jetson_leg NPY and emit the disjoint remainder.

Why
---
The foundation checkpoints under saved/pretrain_base_merged_* were domain-adapted
on the merged camargo/molinaro/scherpereel sources and have never seen jetson
data, while every jetson window has so far been handed to the classifier WITH its
label. That confounds two different things: what the extra data is worth, and what
the extra data is worth *as unlabeled data*.

This script sets up the data-matched two-arm ablation that separates them. It
takes S seconds per (subject, class) out of a source NPY as an SSL-only set and
writes the complement as the classifier's training pool, so both arms can be
trained on the SAME labelled windows and differ only in whether the carved
windows were consumed by continued pretraining (pretrain_dapt.py).

Window accounting
-----------------
jetson_leg.py windows each activity segment with a plain reshape, so windows are
NON-overlapping, and `--leg both` writes Left and Right as separate samples drawn
from the same instant. One window is therefore 2 s of signal but only 1 s of
wall-clock recording, and "30 seconds per class" means 30 windows.

Sampling
--------
Windows are drawn at RANDOM within each (subject, class) cell, never as the first
N. The NPY is in trial order and jetson_leg.discover_trials walks positions in
sorted order -- Leg before Pocket -- so a head slice of a mount-pooled source
would land entirely on leg-mounted trials. The pooled NPY carries no mount
column (label is [activity_id, user_id]), so proportional mount coverage can only
be had statistically; the printed index-position diagnostic is the proxy for it.

Usage (from the repo root)
--------------------------
    python dataset/carve_ssl_split.py --seconds_per_class 30 --seed 3431
    python dataset/carve_ssl_split.py --fraction 0.33 --seed 3431

Writes data_/label_<version>_{ssl<S>,rest<S>}.npy next to the source, plus a
carve_index_<version>_ssl<S>.npz holding both index arrays, and prints the two
dataset/data_config.json entries to paste in (same print-and-paste convention as
jetson_leg.py -- rewriting that JSON programmatically would reorder every entry).
"""

import argparse
import json
import os

import numpy as np

DATA_DIR = os.path.join('dataset', 'jetson_leg')
DATASET = 'jetson_leg'
VERSION = '10_20_both_01030405_xyz_both'
CONFIG_PATH = os.path.join('dataset', 'data_config.json')


def load_source(data_dir, version):
    data = np.load(os.path.join(data_dir, f'data_{version}.npy'))
    label = np.load(os.path.join(data_dir, f'label_{version}.npy'))
    if data.shape[0] != label.shape[0]:
        raise SystemExit(f'{version}: data has {data.shape[0]} windows but label has '
                         f'{label.shape[0]}')
    if label.ndim != 3 or label.shape[2] < 2:
        raise SystemExit(f'{version}: label shape {label.shape} is not (N, seq_len, >=2); '
                         f'this script needs [activity_id, user_id]')
    return data, label


def choose_indices(cls, usr, n_per_cell, rng):
    """-> (ssl_idx, rest_idx, report). n_per_cell(available) gives the cell budget."""
    ssl_parts, report = [], []
    for u in sorted(np.unique(usr)):
        for c in sorted(np.unique(cls)):
            cell = np.flatnonzero((cls == c) & (usr == u))
            want = n_per_cell(cell.size)
            take = min(want, cell.size)
            if take:
                ssl_parts.append(rng.permutation(cell)[:take])
            report.append({'subject': int(u), 'cls': int(c), 'available': int(cell.size),
                           'want': int(want), 'taken': int(take)})
    ssl_idx = np.sort(np.concatenate(ssl_parts)) if ssl_parts else np.empty(0, dtype=int)
    mask = np.ones(cls.size, dtype=bool)
    mask[ssl_idx] = False
    return ssl_idx, np.flatnonzero(mask), report


def check_split(ssl_idx, rest_idx, n_total, label, ssl_label, rest_label, names):
    """Every invariant the two arms rest on. Aborts rather than writing a bad split."""
    assert np.intersect1d(ssl_idx, rest_idx).size == 0, 'ssl and rest indices overlap'
    assert ssl_idx.size + rest_idx.size == n_total, 'ssl + rest does not cover the source'
    assert np.array_equal(np.union1d(ssl_idx, rest_idx), np.arange(n_total)), \
        'ssl u rest is not exactly the source index range'
    # utils.partition_and_reshape shifts labels by the GLOBAL min of the array it is
    # given, so a split that loses class 0 would silently renumber every class.
    for tag, lab in (('ssl', ssl_label), ('rest', rest_label)):
        got = int(lab[:, :, 0].min())
        assert got == 0, (f'{tag}: lowest activity id is {got}, not 0 -- '
                          f'partition_and_reshape would renumber the classes')
    src_classes = set(np.unique(label[:, 0, 0]).astype(int).tolist())
    rest_classes = set(np.unique(rest_label[:, 0, 0]).astype(int).tolist())
    missing = sorted(src_classes - rest_classes)
    assert not missing, ('rest lost classes ' +
                         ', '.join(f'{i}:{names[i]}' for i in missing) +
                         ' -- the carve-out is too large')


def config_entry(dataset, version, tag, size, source_entry):
    entry = dict(source_entry)
    entry['size'] = int(size)
    return {f'{dataset}_{version}_{tag}': entry}


def main():
    ap = argparse.ArgumentParser(
        description='Carve an unlabeled SSL set out of a jetson_leg NPY')
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--version', default=VERSION, help='source dataset_version')
    ap.add_argument('--seconds_per_class', type=int, default=None,
                    help='windows to carve per (subject, class); 1 window = 1 s of '
                         'wall-clock recording. Mutually exclusive with --fraction.')
    ap.add_argument('--fraction', type=float, default=None,
                    help='carve this fraction of every (subject, class) instead of a '
                         'fixed count -- gentler on the small transition classes.')
    ap.add_argument('--seed', type=int, default=3431)
    ap.add_argument('--config', default=CONFIG_PATH)
    ap.add_argument('--dry', action='store_true', help='report the split, write nothing')
    args = ap.parse_args()

    if (args.seconds_per_class is None) == (args.fraction is None):
        raise SystemExit('pass exactly one of --seconds_per_class or --fraction')
    if args.fraction is not None:
        if not 0 < args.fraction < 1:
            raise SystemExit(f'--fraction must be in (0, 1), got {args.fraction}')
        n_per_cell = lambda n: int(np.floor(n * args.fraction))  # noqa: E731
        ssl_tag = f'sslf{int(round(args.fraction * 100))}'
        rest_tag = f'restf{int(round(args.fraction * 100))}'
        budget = f'{args.fraction:g} of every (subject, class)'
    else:
        n_per_cell = lambda n: args.seconds_per_class  # noqa: E731
        ssl_tag = f'ssl{args.seconds_per_class}'
        rest_tag = f'rest{args.seconds_per_class}'
        budget = f'{args.seconds_per_class} s per (subject, class)'

    data, label = load_source(args.data_dir, args.version)
    cls = label[:, 0, 0].astype(int)
    usr = label[:, 0, 1].astype(int)

    all_config = json.load(open(args.config, 'r'))
    source_key = f'{args.dataset}_{args.version}'
    if source_key not in all_config:
        raise SystemExit(f'{source_key} is not in {args.config}')
    source_entry = all_config[source_key]
    names = list(source_entry.get('activity_label', []))
    subjects = list(source_entry.get('user_label', []))
    if source_entry.get('size') != data.shape[0]:
        raise SystemExit(f'{source_key}: config says size={source_entry.get("size")} but the '
                         f'NPY holds {data.shape[0]} windows -- fix the config first')

    rng = np.random.default_rng(args.seed)
    ssl_idx, rest_idx, report = choose_indices(cls, usr, n_per_cell, rng)
    ssl_label, rest_label = label[ssl_idx], label[rest_idx]
    check_split(ssl_idx, rest_idx, data.shape[0], label, ssl_label, rest_label, names)

    print(f'source {source_key}: {data.shape[0]} windows, {len(names)} classes, '
          f'{len(subjects)} subjects {subjects}')
    print(f'carve budget: {budget}   seed {args.seed}\n')

    print('windows carved for SSL per class x subject (taken / available):')
    header = ' '.join(f'{s:>11}' for s in subjects) if subjects else ''
    print(f"{'class':>14} {header}{'ssl':>8}{'rest':>8}")
    short = []
    for c in sorted(np.unique(cls)):
        cells = [r for r in report if r['cls'] == c]
        line = ' '.join(f"{r['taken']:>4}/{r['available']:<6}" for r in cells)
        n_ssl = sum(r['taken'] for r in cells)
        n_rest = sum(r['available'] - r['taken'] for r in cells)
        print(f'{names[c]:>14} {line}{n_ssl:>8}{n_rest:>8}')
        short += [(names[c], subjects[r['subject']] if subjects else r['subject'])
                  for r in cells if r['taken'] < r['want']]
    print(f"{'TOTAL':>14} {'':>{max(len(header), 0)}}{ssl_idx.size:>8}{rest_idx.size:>8}")
    if short:
        print(f'\n  WARNING: {len(short)} (class, subject) cells had fewer windows than the '
              f'budget and were taken whole:')
        for name, subj in short:
            print(f'    {name} / {subj}')

    rest_per_class = np.array([int((rest_label[:, 0, 0].astype(int) == c).sum())
                               for c in sorted(np.unique(cls))])
    smallest = int(rest_per_class.min())
    print(f'\nrest: {rest_idx.size} windows, smallest class {smallest} '
          f'-> per-class labelled budget is capped at ~{int(smallest * 0.8)} '
          f'(BALANCE=1 clamps at the smallest class of the 80% train split)')
    print(f'ssl : {ssl_idx.size} windows = {ssl_idx.size / 60:.1f} min of wall-clock recording')

    # Mount-balance proxy: subject blocks are contiguous in the NPY and Leg trials
    # precede Pocket trials inside each block, so a mount-biased draw shows up as a
    # mean normalised index far from 0.5. Diagnostic only -- there is no mount column.
    print('\nmean normalised index of carved windows within each subject '
          '(0.5 = mount-unbiased; Leg trials occupy the low half):')
    for u in sorted(np.unique(usr)):
        block = np.flatnonzero(usr == u)
        lo, hi = block.min(), block.max()
        picked = ssl_idx[(ssl_idx >= lo) & (ssl_idx <= hi)]
        pos = (picked - lo) / max(hi - lo, 1)
        name = subjects[u] if subjects else f'id{u}'
        print(f'  {name}: {pos.mean():.3f}  (n={picked.size})')

    entries = {}
    entries.update(config_entry(args.dataset, args.version, ssl_tag, ssl_idx.size, source_entry))
    entries.update(config_entry(args.dataset, args.version, rest_tag, rest_idx.size, source_entry))

    if args.dry:
        print('\n--dry: nothing written.')
        return

    for tag, idx in ((ssl_tag, ssl_idx), (rest_tag, rest_idx)):
        version = f'{args.version}_{tag}'
        np.save(os.path.join(args.data_dir, f'data_{version}.npy'), data[idx])
        np.save(os.path.join(args.data_dir, f'label_{version}.npy'), label[idx])
        print(f'\nwrote data_{version}.npy / label_{version}.npy  ({idx.size} windows)')
    index_path = os.path.join(args.data_dir, f'carve_index_{args.version}_{ssl_tag}.npz')
    np.savez(index_path, ssl=ssl_idx, rest=rest_idx, seed=args.seed)
    print(f'wrote {index_path}')

    print(f'\nAdd to {args.config}:')
    print(json.dumps(entries, indent=4))


if __name__ == '__main__':
    main()
