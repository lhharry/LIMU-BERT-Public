import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats


DEFAULT_TRAIN_DATA = Path("dataset/motion/data_20_120.npy")
DEFAULT_TRAIN_LABEL = Path("dataset/motion/label_20_120.npy")


MOTION_HEADERS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def load_training_data(data_path: Path) -> np.ndarray:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    data = np.load(data_path)
    if data.ndim != 3:
        raise ValueError(f"Expected training data shape (N, W, F), got {data.shape}")
    return data.astype(np.float32)


def load_training_label(label_path: Path | None) -> np.ndarray | None:
    if label_path is None or not label_path.exists():
        return None

    labels = np.load(label_path)
    if labels.ndim != 3:
        raise ValueError(f"Expected label shape (N, W, L), got {labels.shape}")
    return labels


def parse_feature_columns(columns_arg: str | None) -> list[str] | None:
    if not columns_arg:
        return None
    columns = [column.strip() for column in columns_arg.split(",") if column.strip()]
    return columns or None


def resolve_activity_index(activity_name: str, label_names: list[str] | None) -> int:
    if label_names is None:
        raise ValueError("Dataset config does not provide activity label names.")

    if activity_name.isdigit():
        activity_index = int(activity_name)
        if 0 <= activity_index < len(label_names):
            return activity_index
        raise ValueError(f"Activity index out of range: {activity_index}")

    if activity_name not in label_names:
        raise ValueError(f"Unknown activity '{activity_name}'. Available: {label_names}")
    return label_names.index(activity_name)


def find_contiguous_activity_run(labels: np.ndarray, activity_index: int) -> tuple[int, int]:
    activity_series = labels[:, 0, 0].astype(int)
    matches = np.where(activity_series == activity_index)[0]
    if matches.size == 0:
        raise ValueError(f"No samples found for activity index {activity_index}")

    run_start = matches[0]
    run_end = matches[0]
    for idx in matches[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            break
    return run_start, run_end + 1


def make_continuous_snippet(data: np.ndarray, start: int, end: int, max_windows: int | None = None) -> np.ndarray:
    snippet = data[start:end]
    if snippet.size == 0:
        raise ValueError("Snippet slice is empty")
    if max_windows is not None:
        snippet = snippet[:max_windows]
    return np.concatenate(snippet, axis=0).astype(np.float32)


def load_csv_window(csv_path: Path, delimiter: str, columns: list[str], window_size: int) -> tuple[np.ndarray, list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path, sep=delimiter)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if len(df) < window_size:
        raise ValueError(f"CSV has only {len(df)} rows, but window_size={window_size} requires at least that many.")

    window = df[columns].iloc[:window_size].to_numpy(dtype=np.float32)
    return window, list(df.columns)


def infer_headers(dataset_name: str, feature_count: int) -> list[str]:
    if dataset_name in {"motion", "hhar", "uci"} and feature_count == 6:
        return MOTION_HEADERS
    if dataset_name == "shoaib" and feature_count == 9:
        return [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
        ]
    return [f"feature_{i}" for i in range(feature_count)]


def plot_channels(ax, window: np.ndarray, headers: list[str], channel_indices: list[int], title: str, color: str, label_prefix: str) -> None:
    steps = np.arange(window.shape[0])
    line_styles = ["-", "--", ":", "-.", "-", "--"]
    for plot_idx, channel_idx in enumerate(channel_indices):
        ax.plot(
            steps,
            window[:, channel_idx],
            color=color,
            linestyle=line_styles[plot_idx % len(line_styles)],
            linewidth=1.2,
            label=f"{label_prefix} {headers[channel_idx]}",
        )
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training IMU windows and compare them with a CSV log.")
    parser.add_argument("--dataset", type=str, default="motion", choices=["hhar", "motion", "uci", "shoaib"])
    parser.add_argument("--version", type=str, default="20_120", choices=["10_100", "20_120"])
    parser.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--train-label", type=str, default=str(DEFAULT_TRAIN_LABEL))
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV file to compare against training data")
    parser.add_argument("--delimiter", type=str, default=";", help="CSV delimiter")
    parser.add_argument("--csv-columns", type=str, default=None, help="Comma-separated CSV columns to plot in order")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of training window to plot")
    parser.add_argument("--activity", type=str, default="walking", help="Optional activity name or index for a continuous snippet, e.g. walking")
    parser.add_argument("--snippet-windows", type=int, default=5, help="Number of consecutive windows to concatenate when --activity is set")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    dataset_cfg = load_dataset_stats(args.dataset, args.version)
    if dataset_cfg is None:
        raise ValueError(f"Unable to load dataset config for {args.dataset}_{args.version}")

    train_data = load_training_data(Path(args.train_data))
    train_label = load_training_label(Path(args.train_label))
    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)

    if train_data.shape[0] == 0:
        raise ValueError("Training data is empty")
    if train_label is not None and train_label.shape[0] != train_data.shape[0]:
        raise ValueError(f"Data and label sample counts do not match: {train_data.shape[0]} vs {train_label.shape[0]}")

    plot_title = f"Training sample #{args.sample_index}"
    if args.activity:
        if train_label is None:
            raise ValueError("--activity requires --train-label to be present")
        activity_index = resolve_activity_index(args.activity, label_names)
        run_start, run_end = find_contiguous_activity_run(train_label, activity_index)
        run_length = run_end - run_start
        if args.snippet_windows <= 0:
            raise ValueError("--snippet-windows must be positive")
        snippet_windows = min(args.snippet_windows, run_length)
        if snippet_windows == 0:
            raise ValueError(f"No contiguous windows available for activity {args.activity}")
        snippet_start = run_start
        snippet_end = run_start + snippet_windows
        train_window = make_continuous_snippet(train_data, snippet_start, snippet_end)
        plot_title = f"Continuous {args.activity} snippet: windows {snippet_start}..{snippet_end - 1}"
        print(f"Selected activity: {args.activity} -> index {activity_index}")
        print(f"Continuous run: {run_start}..{run_end - 1} ({run_length} windows)")
        print(f"Using snippet windows: {snippet_start}..{snippet_end - 1} ({snippet_windows} windows)")
    else:
        if not (0 <= args.sample_index < train_data.shape[0]):
            raise IndexError(f"sample_index must be in [0, {train_data.shape[0] - 1}]")
        train_window = train_data[args.sample_index]

    feature_count = train_window.shape[1]
    train_headers = infer_headers(args.dataset, feature_count)

    print("=== Training Data Info ===")
    print(f"Dataset: {args.dataset}_{args.version}")
    print(f"Training data path: {args.train_data}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Feature count: {feature_count}")
    print("Training headers:")
    for i, header in enumerate(train_headers[:feature_count]):
        print(f"  {i}: {header}")
    if label_names:
        print(f"Label names: {label_names}")
    else:
        print(f"Label count: {label_num}")

    if train_label is not None and not args.activity:
        sample_label = train_label[args.sample_index, 0, dataset_cfg.activity_label_index]
        if label_names and 0 <= int(sample_label) < len(label_names):
            sample_label_name = label_names[int(sample_label)]
        else:
            sample_label_name = str(int(sample_label))
        print(f"Sample label at index {args.sample_index}: {sample_label} ({sample_label_name})")

    csv_window = None
    csv_headers = None
    if args.csv_path:
        csv_columns = parse_feature_columns(args.csv_columns)
        if csv_columns is None:
            csv_columns = infer_headers(args.dataset, feature_count)
        csv_window, csv_headers = load_csv_window(Path(args.csv_path), args.delimiter, csv_columns, train_window.shape[0])
        print("\n=== CSV Comparison Info ===")
        print(f"CSV path: {args.csv_path}")
        print(f"CSV headers available: {list(csv_headers)[:20]}")
        print("CSV plotted columns:")
        for i, header in enumerate(csv_columns):
            print(f"  {i}: {header}")

    accel_indices = [0, 1, 2]
    gyro_indices = [3, 4, 5]
    if feature_count < 6:
        raise ValueError(f"Expected at least 6 features for accel+gyro plotting, got {feature_count}")

    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    plot_channels(
        ax_acc,
        train_window,
        train_headers,
        accel_indices,
        f"{plot_title} - Accelerometer",
        color="tab:blue",
        label_prefix="train",
    )
    plot_channels(
        ax_gyro,
        train_window,
        train_headers,
        gyro_indices,
        f"{plot_title} - Gyroscope",
        color="tab:red",
        label_prefix="train",
    )

    if csv_window is not None:
        plot_channels(ax_acc, csv_window, train_headers, accel_indices, f"{plot_title} - Accelerometer", color="tab:orange", label_prefix="csv")
        plot_channels(ax_gyro, csv_window, train_headers, gyro_indices, f"{plot_title} - Gyroscope", color="tab:orange", label_prefix="csv")

    ax_acc.legend(loc="upper right", ncol=3, fontsize=8)
    ax_gyro.legend(loc="upper right", ncol=3, fontsize=8)
    ax_gyro.set_xlabel("time step")
    fig.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()