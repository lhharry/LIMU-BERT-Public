"""
Count label occurrences across all training_data folders.

Reads every CSV in ROOT/ABxx/training_data/ and aggregates the last column
(which is 'Label' for walking/ramp/stair and 'Speed' for treadmill).

Reports:
  - per-subject counts
  - overall totals
  - number of distinct labels
"""

from pathlib import Path
from collections import Counter
import pandas as pd

# ---- EDIT THIS to point at your dataset root ----
ROOT = Path(r"D:\01_Code\DATA\OpenSource\AY_Data")


def count_one_file(csv_path: Path) -> tuple[str, Counter]:
    """Return (label_column_name, Counter of values) for a single CSV."""
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]           # last column is the label/speed
    counts = Counter(df[label_col].dropna().tolist())
    return label_col, counts


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"ROOT not found: {ROOT}")

    overall = Counter()               # total counts across all subjects, Label column
    overall_speed = Counter()         # total counts for Speed column (treadmill)
    per_subject = {}                  # {subject: Counter} for Label column only
    files_seen = 0

    subjects = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name.startswith("AB")
    )

    for subject_dir in subjects:
        td = subject_dir / "training_data"
        if not td.is_dir():
            continue

        subj_counts = Counter()
        for csv_path in sorted(td.glob("*.csv")):
            try:
                label_col, counts = count_one_file(csv_path)
            except Exception as e:
                print(f"error reading {csv_path.name}: {e}")
                continue

            files_seen += 1
            if label_col.lower() == "speed":
                overall_speed.update(counts)
            else:
                subj_counts.update(counts)
                overall.update(counts)

        per_subject[subject_dir.name] = subj_counts

    # ---- report ----
    print(f"Scanned {files_seen} files across {len(per_subject)} subjects.\n")

    print("=== Per-subject label counts (levelground / ramp / stair) ===")
    for subj, counts in per_subject.items():
        if not counts:
            print(f"{subj}: (no data)")
            continue
        parts = ", ".join(f"{lbl}={n}" for lbl, n in sorted(counts.items()))
        print(f"{subj}: {parts}")

    print("\n=== Overall label counts (levelground / ramp / stair) ===")
    print(f"Distinct labels: {len(overall)}")
    for lbl, n in sorted(overall.items(), key=lambda x: -x[1]):
        print(f"  {lbl!r:20s} {n:>10,} rows")
    print(f"Total rows: {sum(overall.values()):,}")

    if overall_speed:
        print("\n=== Treadmill speed counts ===")
        print(f"Distinct speeds: {len(overall_speed)}")
        for spd, n in sorted(overall_speed.items()):
            print(f"  {spd!r:>10} {n:>10,} rows")
        print(f"Total rows: {sum(overall_speed.values()):,}")


if __name__ == "__main__":
    main()
