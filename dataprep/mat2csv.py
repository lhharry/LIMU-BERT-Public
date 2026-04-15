"""
mat2csv.py
Converts a MATLAB .mat file to a flat CSV file.

Default output columns:
  thigh_Accel_X, thigh_Accel_Y, thigh_Accel_Z,
  thigh_Gyro_X, thigh_Gyro_Y, thigh_Gyro_Z,
  label_header, label_id

Usage
-----
  python mat2csv.py subject_AB06.mat
  python mat2csv.py subject_AB06.mat --out_csv ./subject_AB06.csv
  python mat2csv.py subject_AB06.mat --all_data_cols
  python mat2csv.py subject_AB06.mat --no_labels
"""

import argparse
import pathlib
import numpy as np
import scipy.io as sio

SAMPLE_WINDOW = 20
DEFAULT_DATA_COLS = [
    "thigh_Accel_X",
    "thigh_Accel_Y",
    "thigh_Accel_Z",
    "thigh_Gyro_X",
    "thigh_Gyro_Y",
    "thigh_Gyro_Z",
]


def _flatten_cell(cell_array):
    flat = cell_array.flatten()
    result = []
    for item in flat:
        if isinstance(item, np.ndarray):
            result.append(str(item.flat[0]))
        else:
            result.append(str(item))
    return result

def down_sample(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 < data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)


def _infer_hz_from_time(time_s: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).flatten()
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def _align_labels_to_data_time(
    data_time: np.ndarray,
    label_time: np.ndarray,
    label_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align labels to data timestamps by nearest-time lookup.

    This handles cases where label streams are sampled at a higher rate
    than sensor rows in data_arr.
    """
    t_data = np.asarray(data_time, dtype=np.float64).flatten()
    t_label = np.asarray(label_time, dtype=np.float64).flatten()
    y_label = np.asarray(label_id, dtype=np.float64).flatten()

    if t_data.ndim != 1 or t_data.size == 0:
        raise ValueError("data_arr time column must be a non-empty 1D array")
    if t_label.ndim != 1 or y_label.ndim != 1 or t_label.size != y_label.size:
        raise ValueError("label_header and label_id must be 1D arrays of equal length")

    # Sort labels by time for robust nearest-neighbor matching.
    sort_idx = np.argsort(t_label, kind="mergesort")
    t_sorted = t_label[sort_idx]
    y_sorted = y_label[sort_idx]

    # Remove duplicate timestamps while keeping the first occurrence.
    if t_sorted.size > 1:
        keep = np.concatenate(([True], np.diff(t_sorted) != 0))
        t_sorted = t_sorted[keep]
        y_sorted = y_sorted[keep]

    if t_sorted.size == 0:
        raise ValueError("label_header is empty")

    if t_sorted.size == 1:
        aligned_t = np.full_like(t_data, t_sorted[0], dtype=np.float64)
        aligned_y = np.full_like(t_data, y_sorted[0], dtype=np.float64)
        return aligned_t, aligned_y

    pos = np.searchsorted(t_sorted, t_data, side="left")
    pos = np.clip(pos, 1, t_sorted.size - 1)
    left = pos - 1
    right = pos
    choose_right = (t_data - t_sorted[left]) >= (t_sorted[right] - t_data)
    nearest = np.where(choose_right, right, left)

    aligned_t = t_sorted[nearest]
    aligned_y = y_sorted[nearest]
    return aligned_t, aligned_y


def convert_mat_to_csv(
    mat_path: str,
    out_csv: str | None = None,
    all_data_cols: bool = False,
    include_labels: bool = True,
    apply_downsample: bool = True,
    window_target: float | None = None,
    verbose: bool = True,
):
    mat_file = pathlib.Path(mat_path)
    if not mat_file.exists():
        raise FileNotFoundError(f"Cannot find: {mat_file}")

    out_path = pathlib.Path(out_csv) if out_csv else mat_file.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

    if "data_cols" not in mat or "data_arr" not in mat:
        raise KeyError(".mat file must contain 'data_cols' and 'data_arr'")

    col_names = _flatten_cell(mat["data_cols"])
    data_arr = np.asarray(mat["data_arr"], dtype=np.float64)

    if data_arr.ndim != 2:
        raise ValueError(f"Expected data_arr to be 2D, got shape {data_arr.shape}")

    if all_data_cols:
        data_indices = list(range(data_arr.shape[1]))
        data_header = col_names
    else:
        try:
            data_indices = [col_names.index(c) for c in DEFAULT_DATA_COLS]
        except ValueError as e:
            raise ValueError(f"Column not found: {e}\nAvailable: {col_names}")
        data_header = DEFAULT_DATA_COLS

    out_matrix = data_arr[:, data_indices]
    header = list(data_header)

    # data_arr first column is treated as timestamp for optional downsampling
    # and for aligning labels to data rows.
    if data_arr.shape[1] < 1:
        raise ValueError("data_arr must contain at least one column for time alignment")
    data_time = data_arr[:, 0]

    if apply_downsample:
        inferred_hz = _infer_hz_from_time(data_time)
        src_hz = float(window_target) if window_target is not None else inferred_hz
        if src_hz > 0 and src_hz > SAMPLE_WINDOW:
            merged = np.column_stack([data_time, out_matrix])
            merged_ds = down_sample(merged, src_hz)
            data_time = merged_ds[:, 0]
            out_matrix = merged_ds[:, 1:]
            if verbose:
                print(
                    f"[i] Downsampled data using down_sample: {src_hz:.3f} Hz -> {SAMPLE_WINDOW} Hz, "
                    f"rows {data_arr.shape[0]} -> {out_matrix.shape[0]}"
                )
        elif verbose:
            print(
                "[i] Skip down_sample: source rate not above target "
                f"({src_hz:.3f} Hz <= {SAMPLE_WINDOW} Hz)"
            )

    if include_labels:
        if "label_header" not in mat or "label_id" not in mat:
            raise KeyError("include_labels=True but .mat file is missing 'label_header' or 'label_id'")

        label_header = np.asarray(mat["label_header"], dtype=np.float64).flatten()
        label_id = np.asarray(mat["label_id"], dtype=np.float64).flatten()

        # In some datasets, labels are sampled at higher rate than data rows.
        # Align labels to current data timestamps using nearest-time lookup.
        label_header_aligned, label_id_aligned = _align_labels_to_data_time(
            data_time,
            label_header,
            label_id,
        )

        out_matrix = np.column_stack([out_matrix, label_header_aligned, label_id_aligned])
        header.extend(["label_header", "label_id"])

        if verbose and (label_header.shape[0] != out_matrix.shape[0] or label_id.shape[0] != out_matrix.shape[0]):
            abs_dt = np.abs(data_time - label_header_aligned)
            mean_dt = float(np.mean(abs_dt)) if abs_dt.size else 0.0
            max_dt = float(np.max(abs_dt)) if abs_dt.size else 0.0
            print(
                "[i] Label/Data length mismatch detected and aligned by timestamp: "
                f"data={data_time.shape[0]}, label={label_header.shape[0]}, "
                f"mean_abs_dt={mean_dt:.6g}s, max_abs_dt={max_dt:.6g}s"
            )

    np.savetxt(out_path, out_matrix, delimiter=",", header=",".join(header), comments="")

    if verbose:
        print(f"[✓] CSV saved: {out_path}")
        print(f"    shape   : {out_matrix.shape}")
        print(f"    columns : {header}")

    return out_path


def verify(csv_path: str, n_rows: int = 5):
    csv_file = pathlib.Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Cannot find CSV: {csv_file}")

    # Use genfromtxt so we can read header names quickly for a sanity check.
    arr = np.genfromtxt(csv_file, delimiter=",", names=True)
    print("\n--- Verification ---")
    print(f"file        : {csv_file}")
    print(f"rows        : {arr.shape[0]}")
    print(f"columns     : {arr.dtype.names}")
    print(f"first {n_rows} rows:")
    raw = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    print(raw[:n_rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mat to a flat CSV file")
    parser.add_argument("mat_file", help="Path to the input .mat file")
    parser.add_argument("--out_csv", default=None, help="Path to output .csv (default: <mat_file>.csv)")
    parser.add_argument(
        "--all_data_cols",
        action="store_true",
        help="Export all data_arr columns instead of default IMU columns",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not append label_header and label_id columns",
    )
    parser.add_argument(
        "--no_downsample",
        action="store_true",
        help=f"Do not apply down_sample to {SAMPLE_WINDOW} Hz",
    )
    parser.add_argument(
        "--window_target",
        type=float,
        default=None,
        help="Source sample rate used by down_sample (default: infer from data_arr timestamp)",
    )
    parser.add_argument("--verify", action="store_true", help="Reload and print a small verification summary")
    args = parser.parse_args()

    out_csv = convert_mat_to_csv(
        args.mat_file,
        out_csv=args.out_csv,
        all_data_cols=args.all_data_cols,
        include_labels=not args.no_labels,
        apply_downsample=not args.no_downsample,
        window_target=args.window_target,
    )

    if args.verify:
        verify(str(out_csv))
