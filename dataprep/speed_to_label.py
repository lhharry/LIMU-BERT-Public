"""
Convert the treadmill 'Speed' column into a 'Label' column, in-place,
so every training_data CSV uses the same label schema.

Rules:
    Speed == 0    -> 'stand'
    Speed >= 1.4  -> 'jog'
    otherwise     -> 'walk'

The 'Speed' column is dropped and replaced with a 'Label' column at the
end of the file. Files whose last column is already 'Label' are skipped,
so this script is safe to re-run (idempotent).
"""

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# ---- EDIT THIS to point at your dataset root ----
ROOT = Path(r"D:\01_Code\DATA\OpenSource\AY_Data")

JOG_THRESHOLD = 1.4   # m/s


def convert_file(csv_path: Path) -> Counter | None:
    """Rewrite csv_path with Speed -> Label. Returns counts, or None if skipped."""
    df = pd.read_csv(csv_path)
    last_col = df.columns[-1]

    if last_col != "Speed":
        return None  # already converted, or not a treadmill file

    speed = df["Speed"].to_numpy()
    label = np.where(
        speed == 0, "stand",
        np.where(speed >= JOG_THRESHOLD, "jog", "walk"),
    )

    df = df.drop(columns=["Speed"])
    df["Label"] = label   # appended as the new last column

    df.to_csv(csv_path, index=False)
    return Counter(label.tolist())


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"ROOT not found: {ROOT}")

    grand_total = Counter()
    converted = 0
    skipped = 0

    subjects = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name.startswith("AB")
    )

    for subject_dir in subjects:
        td = subject_dir / "training_data"
        if not td.is_dir():
            continue

        subj_total = Counter()
        for csv_path in sorted(td.glob("*.csv")):
            try:
                result = convert_file(csv_path)
            except Exception as e:
                print(f"error on {csv_path.name}: {e}")
                continue

            if result is None:
                skipped += 1
                continue

            converted += 1
            subj_total.update(result)

        if subj_total:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(subj_total.items()))
            print(f"{subject_dir.name}: {parts}")
            grand_total.update(subj_total)

    print(f"\nConverted {converted} file(s), skipped {skipped} (already Label or non-Speed).")
    if grand_total:
        print("\n=== Overall treadmill label counts after conversion ===")
        for lbl, n in sorted(grand_total.items(), key=lambda x: -x[1]):
            print(f"  {lbl!r:10s} {n:>10,} rows")
        print(f"Total rows: {sum(grand_total.values()):,}")


if __name__ == "__main__":
    main()
