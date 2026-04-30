from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path("inference/data/levelground_ccw_normal_01_04.csv")
DEFAULT_CSV_SAVE_PATH = Path("inference/data/")


def down_sample(data, raw_sr, target_sr):
    window_sample = raw_sr * 1.0 / target_sr
    result = []
    if window_sample < 1:
        raise ValueError('target_sr must be less than or equal to raw_sr')
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
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                i += window
    return np.array(result)


def downsample_csv(input_path: Path, output_dir: Path, raw_sr: int, target_sr: int):
    df = pd.read_csv(input_path)

    # Separate timestamp, sensor data, and label
    time_col = df.columns[0]
    label_col = df.columns[-1]
    sensor_cols = df.columns[1:-1]

    sensor_data = df[sensor_cols].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy()

    # Downsample sensor data (numeric averaging)
    sensor_ds = down_sample(sensor_data, raw_sr, target_sr)

    # For label: pick the most common label in each window (mode)
    ratio = raw_sr / target_sr
    label_ds = []
    i = 0
    while i < len(labels):
        window = int(ratio) + (1 if (i / ratio - int(i / ratio)) >= 0.5 else 0)
        window = max(1, min(window, len(labels) - i))
        chunk = labels[i: i + window]
        values, counts = np.unique(chunk, return_counts=True)
        label_ds.append(values[np.argmax(counts)])
        i += window
        if len(label_ds) >= len(sensor_ds):
            break

    # Re-generate timestamps at target_sr
    n = len(sensor_ds)
    start_time = df[time_col].iloc[0]
    step = 1.0 / target_sr
    timestamps = np.round(start_time + np.arange(n) * step, decimals=6)

    df_out = pd.DataFrame(sensor_ds, columns=sensor_cols)
    df_out.insert(0, time_col, timestamps)
    df_out[label_col] = label_ds[:n]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem + f"_ds{target_sr}hz"
    output_path = output_dir / (stem + ".csv")
    df_out.to_csv(output_path, index=False)
    print("Saved downsampled CSV to:", output_path)
    print("  Original samples: %d  →  Downsampled: %d  (%d Hz → %d Hz)"
          % (len(df), n, raw_sr, target_sr))
    return output_path


if __name__ == "__main__":
    input_path = DEFAULT_CSV_PATH
    output_dir = DEFAULT_CSV_SAVE_PATH
    raw_sr = 200
    target_sr = 10

    downsample_csv(input_path, output_dir, raw_sr, target_sr)
