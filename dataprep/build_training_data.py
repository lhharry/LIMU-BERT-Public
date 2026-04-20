"""
Combine IMU + condition CSVs into a single training_data folder per subject.

Folder layout expected:
    ROOT/ABxx/<date>/<activity>/imu/<name>.csv          <- Header + 24 sensor cols
    ROOT/ABxx/<date>/<activity>/conditions/<name>.csv   <- Header + Label (or Speed)

Output:
    ROOT/ABxx/training_data/<name>.csv
    Columns: Header, <all IMU sensor cols>, Label (or Speed)
"""

from pathlib import Path
import pandas as pd

# ---- EDIT THIS to point at your dataset root ----
ROOT = Path(r"D:\01_Code\DATA\OpenSource\AY_Data")

ACTIVITIES = ["levelground", "ramp", "stair", "treadmill"]


def combine_one(imu_path: Path, cond_path: Path, out_path: Path) -> str:
    """Merge a single IMU/condition pair and write the result."""
    imu_df = pd.read_csv(imu_path)
    cond_df = pd.read_csv(cond_path)

    # The label column is whatever isn't 'Header' (Label for walking, Speed for treadmill)
    label_cols = [c for c in cond_df.columns if c != "Header"]
    if not label_cols:
        return f"skip (no label column): {cond_path.name}"
    label_col = label_cols[0]

    # Merge on Header so rows line up even if the two files aren't the exact same length
    merged = imu_df.merge(cond_df[["Header", label_col]], on="Header", how="inner")

    if merged.empty:
        return f"skip (no matching Header rows): {cond_path.name}"

    # Enforce column order: Header first, label last, sensors in between
    middle = [c for c in merged.columns if c not in ("Header", label_col)]
    merged = merged[["Header"] + middle + [label_col]]

    merged.to_csv(out_path, index=False)
    return f"ok  {out_path.name}  ({len(merged)} rows, label='{label_col}')"


def process_subject(subject_dir: Path) -> None:
    # Find the date folder inside ABxx (there's only one, and skip our own output folder)
    date_dirs = [
        d for d in subject_dir.iterdir()
        if d.is_dir() and d.name != "training_data"
    ]
    if not date_dirs:
        print(f"  no date folder found in {subject_dir.name}")
        return
    date_dir = date_dirs[0]

    out_dir = subject_dir / "training_data"
    out_dir.mkdir(exist_ok=True)

    for activity in ACTIVITIES:
        cond_dir = date_dir / activity / "conditions"
        imu_dir = date_dir / activity / "imu"
        if not (cond_dir.is_dir() and imu_dir.is_dir()):
            print(f"  [{activity}] folders missing, skipping")
            continue

        cond_files = sorted(cond_dir.glob("*.csv"))
        if not cond_files:
            print(f"  [{activity}] no condition CSVs")
            continue

        for cond_file in cond_files:
            imu_file = imu_dir / cond_file.name
            if not imu_file.exists():
                print(f"  [{activity}] no IMU match for {cond_file.name}")
                continue

            out_path = out_dir / cond_file.name  # filename already contains activity prefix
            try:
                msg = combine_one(imu_file, cond_file, out_path)
            except Exception as e:
                msg = f"error on {cond_file.name}: {e}"
            print(f"  [{activity}] {msg}")


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"ROOT not found: {ROOT}")

    subjects = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name.startswith("AB")
    )
    if not subjects:
        raise SystemExit(f"No ABxx folders found in {ROOT}")

    for subject_dir in subjects:
        print(f"\n=== {subject_dir.name} ===")
        process_subject(subject_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
