"""
mat2npy.py
Converts a MATLAB .mat file to two windowed .npy files:

  data_<step>_<win>.npy    shape: (N_windows, window_size, 6)   float64
      channels: thigh_Accel_X/Y/Z, thigh_Gyro_X/Y/Z

  label_<step>_<win>.npy   shape: (N_windows, window_size, 2)   float64
      channels: [label_header, label_id]
      label_header is timestamp and is downsampled to target_hz before windowing

Usage
-----
  python mat2npy.py subject_AB06.mat
  python mat2npy.py subject_AB06.mat --window 120 --step 20
    python mat2npy.py subject_AB06.mat --target_hz 20
  python mat2npy.py subject_AB06.mat --window 120 --step 20 --out_dir ./npy_data
  python mat2npy.py subject_AB06.mat --verify

Label Explain:
1: 'idle'
2: 'walk-stairascent'
3: 'stairascent'
4: 'stairascent-walk'
5: 'walk-stairdescent'
6: 'stairdescent'
7: 'stairdescent-walk'

"downstairs", "upstairs", "sitting", "standing", "walking", "jogging"
"""

import argparse
import pathlib
import numpy as np
import scipy.io as sio

# ── Columns to extract from data_arr ──────────────────────────────────────── #
DATA_COLS = [
    "thigh_Accel_X",
    "thigh_Accel_Y",
    "thigh_Accel_Z",
    "thigh_Gyro_X",
    "thigh_Gyro_Y",
    "thigh_Gyro_Z",
]
# Label channels saved in label file
LABEL_COLS = ["label_header", "label_id"]

# Default sliding-window parameters (matches reference files data_20_120 / label_20_120)
DEFAULT_WINDOW = 120
DEFAULT_STEP   = 20
DEFAULT_TARGET_HZ = 20.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _flatten_cell(cell_array):
    flat = cell_array.flatten()
    result = []
    for item in flat:
        if isinstance(item, np.ndarray):
            result.append(str(item.flat[0]))
        else:
            result.append(str(item))
    return result


def sliding_windows(signal: np.ndarray, window: int, step: int) -> np.ndarray:
    """
    Sliding window over axis-0.

    Parameters
    ----------
    signal : (T, C)
    window : window length in samples
    step   : hop size in samples

    Returns
    -------
    (N_windows, window, C)
    """
    T = signal.shape[0]
    starts = range(0, T - window + 1, step)
    return np.stack([signal[s: s + window] for s in starts], axis=0)


def _build_resample_indices_from_time(time_s: np.ndarray, target_hz: float) -> np.ndarray:
    """
    Build index mapping that resamples a time series to target_hz using nearest points.
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be > 0")

    t = np.asarray(time_s, dtype=np.float64).flatten()
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time header must be a 1D array with at least 2 values")

    # Keep only strictly increasing timestamps to avoid ambiguous nearest lookups.
    valid = np.concatenate(([True], np.diff(t) > 0))
    t_clean = t[valid]
    idx_clean = np.nonzero(valid)[0]
    if t_clean.size < 2:
        raise ValueError("time header does not contain enough increasing samples")

    dt_target = 1.0 / target_hz
    t_start = t_clean[0]
    t_end = t_clean[-1]
    target_t = np.arange(t_start, t_end + 0.5 * dt_target, dt_target)

    pos = np.searchsorted(t_clean, target_t, side="left")
    pos = np.clip(pos, 1, t_clean.size - 1)
    left = pos - 1
    right = pos
    choose_right = (target_t - t_clean[left]) >= (t_clean[right] - target_t)
    nearest = np.where(choose_right, right, left)

    mapped_indices = idx_clean[nearest]
    mapped_indices = mapped_indices[np.concatenate(([True], np.diff(mapped_indices) != 0))]
    return mapped_indices


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------

def convert(
    mat_path: str,
    out_dir:  str | None = None,
    window:   int = DEFAULT_WINDOW,
    step:     int = DEFAULT_STEP,
    target_hz: float = DEFAULT_TARGET_HZ,
    verbose:  bool = True,
):
    mat_path = pathlib.Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Cannot find: {mat_path}")

    out_dir = pathlib.Path(out_dir) if out_dir else mat_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load .mat
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    # ── 1. Resolve column indices ─────────────────────────────────────────── #
    col_names = _flatten_cell(mat["data_cols"])

    try:
        data_indices = [col_names.index(c) for c in DATA_COLS]
    except ValueError as e:
        raise ValueError(f"Column not found: {e}\nAvailable: {col_names}")

    # ── 2. Extract raw signals (T, C) ────────────────────────────────────── #
    data_arr     = np.array(mat["data_arr"],     dtype=np.float64)
    raw_data     = data_arr[:, data_indices]                             # (T, 6)

    label_header = np.array(mat["label_header"], dtype=np.float64).flatten()
    label_id     = np.array(mat["label_id"],     dtype=np.float64).flatten()
    raw_label    = np.column_stack([label_header, label_id])             # (T, 2)

    if raw_data.shape[0] != raw_label.shape[0]:
        raise ValueError(
            f"Length mismatch: data has {raw_data.shape[0]} rows, label has {raw_label.shape[0]} rows"
        )

    # Downsample using timestamp header so all channels stay time-aligned at target_hz.
    ds_idx = _build_resample_indices_from_time(label_header, target_hz)
    data_ds = raw_data[ds_idx]
    label_ds = raw_label[ds_idx]

    T_raw = raw_data.shape[0]
    T = data_ds.shape[0]
    n_wins = len(range(0, T - window + 1, step))
    if verbose:
        duration = float(label_ds[-1, 0] - label_ds[0, 0]) if T > 1 else 0.0
        est_hz = (T - 1) / duration if duration > 0 else 0.0
        print(f"[i] Raw signal  : {T_raw} samples")
        print(f"[i] Downsampled : {T} samples @ ~{est_hz:.3f} Hz (target {target_hz:g} Hz)")
        print(f"[i] Window={window}, Step={step}  →  {n_wins} windows")

    # ── 3. Sliding windows ───────────────────────────────────────────────── #
    data_win  = sliding_windows(data_ds,  window, step)   # (N, 120, 6)
    label_win = sliding_windows(label_ds, window, step)   # (N, 120, 2)

    # ── 4. Save ──────────────────────────────────────────────────────────── #
    tag        = f"{step}_{window}"
    data_path  = out_dir / f"data_{tag}.npy"
    label_path = out_dir / f"label_{tag}.npy"

    np.save(str(data_path),  data_win)
    np.save(str(label_path), label_win)

    if verbose:
        print(f"\n[✓] data  → {data_path}")
        print(f"    shape    : {data_win.shape}  dtype: {data_win.dtype}")
        print(f"    channels : {DATA_COLS}")

        print(f"\n[✓] label → {label_path}")
        print(f"    shape    : {label_win.shape}  dtype: {label_win.dtype}")
        print(f"    channels : {LABEL_COLS}")
        print(f"    unique label_id : {np.unique(label_win[:, :, 1])}")

    return data_path, label_path


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(data_path: str, label_path: str):
    print("\n--- Verification ---")
    d = np.load(data_path)
    l = np.load(label_path)
    print(f"data  shape : {d.shape}   dtype: {d.dtype}")
    print(f"label shape : {l.shape}   dtype: {l.dtype}")
    print(f"\ndata  [win 0, rows 0:3]:\n{d[0, :3]}")
    print(f"\nlabel [win 0, rows 0:3]:\n{l[0, :3]}")
    print(f"\nUnique label_id : {np.unique(l[:, :, 1])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .mat → data_<step>_<win>.npy  +  label_<step>_<win>.npy"
    )
    parser.add_argument("mat_file",           help="Path to the input .mat file")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Window size in samples (default: {DEFAULT_WINDOW})")
    parser.add_argument("--step",   type=int, default=DEFAULT_STEP,
                        help=f"Sliding step in samples (default: {DEFAULT_STEP})")
    parser.add_argument("--target_hz", type=float, default=DEFAULT_TARGET_HZ,
                        help=f"Target sampling rate for timestamp-based downsampling (default: {DEFAULT_TARGET_HZ:g} Hz)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same folder as .mat file)")
    parser.add_argument("--verify", action="store_true",
                        help="Re-load and print a summary after saving")
    args = parser.parse_args()

    d_path, l_path = convert(
        args.mat_file,
        out_dir=args.out_dir,
        window=args.window,
        step=args.step,
        target_hz=args.target_hz,
    )

    if args.verify:
        verify(str(d_path), str(l_path))