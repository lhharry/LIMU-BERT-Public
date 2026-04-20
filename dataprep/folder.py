"""
Safely remove .mat files and resulting empty folders.

Defaults to DRY-RUN — you must pass -y / --confirm to actually delete.

Usage examples:

    # preview what would be deleted (nothing is actually removed)
    python cleanup_mat.py --root "D:/01_Code/DATA/OpenSource/AY Data"

    # same but only inside imu/ and conditions/ sensor folders
    python cleanup_mat.py --root "D:/..." --sensors imu conditions

    # only delete .mat files that have a same-named .csv next to them
    # (safe — guarantees no data loss if conversion didn't run)
    python cleanup_mat.py --root "D:/..." --require_csv

    # ACTUALLY DELETE (combine with any of the above)
    python cleanup_mat.py --root "D:/..." --require_csv -y
"""

import argparse
import os
from pathlib import Path


def find_mats(root: Path, sensors):
    """Yield every .mat file under root. If sensors is given, only yield .mats
    whose parent folder name is in sensors."""
    for p in root.rglob('*.mat'):
        if sensors and p.parent.name not in sensors:
            continue
        yield p


def has_sibling_csv(mat_path: Path) -> bool:
    return mat_path.with_suffix('.csv').exists()


def remove_empty_dirs(root: Path, confirm: bool):
    """Walk bottom-up, remove any directory that ends up empty."""
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        if d == root:
            continue
        try:
            if not any(d.iterdir()):
                if confirm:
                    d.rmdir()
                removed.append(d)
        except OSError:
            pass
    return removed


def fmt_size(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--root', required=True, help='Folder to clean up (searched recursively)')
    p.add_argument('--sensors', nargs='*', default=None,
                   help='Only delete .mat files inside folders with these names '
                        '(e.g. imu conditions). Default: any folder.')
    p.add_argument('--require_csv', action='store_true',
                   help='Only delete a .mat if a same-named .csv exists next to it. '
                        'Safer — proves the conversion succeeded.')
    p.add_argument('--keep_dirs', action='store_true',
                   help='Do not remove empty directories after deleting .mats.')
    p.add_argument('-y', '--confirm', action='store_true',
                   help='Actually delete (default is dry-run: show what would happen).')
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    mode = 'DELETE' if args.confirm else 'DRY-RUN (nothing will be deleted)'
    print(f"Mode       : {mode}")
    print(f"Root       : {root}")
    print(f"Sensors    : {args.sensors or 'any folder'}")
    print(f"Require CSV: {args.require_csv}")
    print()

    to_delete   = []
    skipped_csv = []
    total_bytes = 0

    for mat in find_mats(root, set(args.sensors) if args.sensors else None):
        if args.require_csv and not has_sibling_csv(mat):
            skipped_csv.append(mat)
            continue
        try:
            total_bytes += mat.stat().st_size
        except OSError:
            pass
        to_delete.append(mat)

    print(f"Found {len(to_delete)} .mat file(s) targeted for deletion "
          f"({fmt_size(total_bytes)} total).")
    if skipped_csv:
        print(f"Skipped {len(skipped_csv)} .mat file(s) with no matching .csv "
              f"(use without --require_csv to include them).")

    # Preview a few
    for m in to_delete[:5]:
        print(f"  [-] {m.relative_to(root)}")
    if len(to_delete) > 5:
        print(f"  ... and {len(to_delete) - 5} more")

    # Execute
    if args.confirm:
        errors = 0
        for mat in to_delete:
            try:
                mat.unlink()
            except OSError as e:
                errors += 1
                print(f"  FAIL {mat}: {e}")
        print(f"\nDeleted {len(to_delete) - errors} / {len(to_delete)} .mat file(s).")
    else:
        print("\n(dry-run: nothing deleted. Re-run with -y to actually delete.)")

    # Empty directories
    if not args.keep_dirs:
        removed_dirs = remove_empty_dirs(root, args.confirm)
        if removed_dirs:
            action = "Removed" if args.confirm else "Would remove"
            print(f"\n{action} {len(removed_dirs)} empty director(ies):")
            for d in removed_dirs[:10]:
                print(f"  [-] {d.relative_to(root)}")
            if len(removed_dirs) > 10:
                print(f"  ... and {len(removed_dirs) - 10} more")


if __name__ == '__main__':
    main()