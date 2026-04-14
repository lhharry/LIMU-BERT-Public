import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import LIMUBertModel4Pretrain, fetch_classifier
from utils import Preprocess4Normalization


DEFAULT_CSV_PATH = Path("imu_log_300s_corrected.csv")
DEFAULT_PRETRAIN_MODEL = Path("saved/pretrain_base_motion_20_120/motion.pt")
DEFAULT_CLASSIFIER_MODEL = Path("saved/classifier_base_gru_motion_20_120/motion.pt")


def parse_feature_columns(columns_arg: str | None) -> list[str] | None:
    if not columns_arg:
        return None
    cols = [c.strip() for c in columns_arg.split(",") if c.strip()]
    return cols or None


def resolve_default_columns(sensor: str, feature_count: int) -> list[str] | None:
    if feature_count != 6:
        return None

    if sensor == "left":
        return [
            "left_accel_x",
            "left_accel_y",
            "left_accel_z",
            "left_gyro_x",
            "left_gyro_y",
            "left_gyro_z",
        ]
    if sensor == "right":
        return [
            "right_accel_x",
            "right_accel_y",
            "right_accel_z",
            "right_gyro_x",
            "right_gyro_y",
            "right_gyro_z",
        ]

    # Generic single-IMU fallback names.
    return ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def load_csv_features(
    csv_path: Path,
    delimiter: str,
    sensor: str,
    feature_count: int,
    feature_columns: list[str] | None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=delimiter)
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing requested feature columns: {missing}")
        values = df[feature_columns].to_numpy(dtype=np.float32)
        if values.shape[1] != feature_count:
            raise ValueError(
                f"Expected {feature_count} feature columns, got {values.shape[1]}: {feature_columns}"
            )
        return values, df, feature_columns

    if sensor == "average":
        required_pairs = [
            ("left_accel_x", "right_accel_x"),
            ("left_accel_y", "right_accel_y"),
            ("left_accel_z", "right_accel_z"),
            ("left_gyro_x", "right_gyro_x"),
            ("left_gyro_y", "right_gyro_y"),
            ("left_gyro_z", "right_gyro_z"),
        ]
        missing = [name for pair in required_pairs for name in pair if name not in df.columns]
        if missing:
            raise ValueError(
                "Sensor='average' requires left/right columns for all accel/gyro axes. "
                f"Missing: {sorted(set(missing))}"
            )
        values = np.stack([
            (df[left].to_numpy(dtype=np.float32) + df[right].to_numpy(dtype=np.float32)) * 0.5
            for left, right in required_pairs
        ], axis=1)
        cols_used = [f"avg({left},{right})" for left, right in required_pairs]
        return values, df, cols_used

    resolved_cols = resolve_default_columns(sensor, feature_count)
    if resolved_cols is None:
        raise ValueError(
            "Unable to infer default columns for this setup. "
            "Provide --feature-columns explicitly."
        )

    missing = [c for c in resolved_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing default columns for sensor='{sensor}': {missing}. "
            "Provide --feature-columns to map your CSV columns manually."
        )

    values = df[resolved_cols].to_numpy(dtype=np.float32)
    return values, df, resolved_cols


def window_features(features: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (T, F), got {features.shape}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if features.shape[0] < window_size:
        raise ValueError(
            f"Not enough rows for one window: rows={features.shape[0]}, window_size={window_size}"
        )

    windows = []
    for start in range(0, features.shape[0] - window_size + 1, stride):
        windows.append(features[start : start + window_size])

    if not windows:
        raise ValueError("No windows were created. Check --window-size and --stride.")

    return np.stack(windows, axis=0).astype(np.float32)


def normalize_sequence_data(data: np.ndarray, feature_count: int) -> np.ndarray:
    normalizer = Preprocess4Normalization(feature_count)
    return np.stack([normalizer(sample) for sample in data], axis=0).astype(np.float32)


def build_models(
    device: torch.device,
    label_num: int,
    classifier_version: str,
    pretrain_model_path: Path,
    classifier_model_path: Path,
):
    pretrain_cfg = load_model_config("pretrain_base", "base", "v1")
    classifier_cfg = load_model_config("classifier_base_gru", "gru", classifier_version)
    if pretrain_cfg is None:
        raise ValueError("Unable to load pretrain model config for base_v1.")
    if classifier_cfg is None:
        raise ValueError(f"Unable to load classifier model config for gru_{classifier_version}.")

    pretrain_model = LIMUBertModel4Pretrain(pretrain_cfg, output_embed=True).to(device)
    classifier_model = fetch_classifier("gru", classifier_cfg, input=pretrain_cfg.hidden, output=label_num).to(device)

    if not pretrain_model_path.exists():
        raise FileNotFoundError(f"Pretrain model not found: {pretrain_model_path}")
    if not classifier_model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {classifier_model_path}")

    pretrain_state = torch.load(pretrain_model_path, map_location=device)
    classifier_state = torch.load(classifier_model_path, map_location=device)
    pretrain_model.load_state_dict(pretrain_state)
    classifier_model.load_state_dict(classifier_state)

    pretrain_model.eval()
    classifier_model.eval()
    return pretrain_model, classifier_model, pretrain_cfg


def predict(data: np.ndarray, pretrain_model, classifier_model, batch_size: int, device: torch.device) -> np.ndarray:
    data_loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            embeddings = pretrain_model(batch)
            logits = classifier_model(embeddings)
            predictions.append(torch.argmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(predictions, axis=0)


def load_labels(label_path: Path | None) -> np.ndarray | None:
    if label_path is None:
        return None
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels = np.load(label_path)
    if labels.ndim == 3:
        return labels[:, 0, 0].astype(np.int64)
    if labels.ndim == 1:
        return labels.astype(np.int64)
    raise ValueError(f"Unsupported label shape: {labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict activity labels from IMU CSV data.")
    parser.add_argument("csv_path", nargs="?", default=str(DEFAULT_CSV_PATH), help="Path to IMU CSV file")
    parser.add_argument("--dataset", type=str, default="motion", choices=["hhar", "motion", "uci", "shoaib"])
    parser.add_argument("--dataset-version", type=str, default="20_120", choices=["10_100", "20_120"])
    parser.add_argument("--delimiter", type=str, default=";", help="CSV delimiter")
    parser.add_argument(
        "--sensor",
        type=str,
        default="left",
        choices=["left", "right", "average", "generic"],
        help="Sensor columns to use when --feature-columns is not provided",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Comma-separated feature columns in desired order (overrides --sensor)",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Sliding window size; default uses model seq_len")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride; default equals window_size")
    parser.add_argument("--feature-count", type=int, default=6, help="Number of sensor features expected by model")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--classifier-version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--pretrain-model", type=str, default=str(DEFAULT_PRETRAIN_MODEL))
    parser.add_argument("--classifier-model", type=str, default=str(DEFAULT_CLASSIFIER_MODEL))
    parser.add_argument("--label-path", type=str, default=None, help="Optional .npy labels for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save predictions (.npy)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        raise ValueError(f"Unable to load dataset config: {args.dataset}_{args.dataset_version}")

    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    if label_num <= 0:
        raise ValueError("Unable to resolve number of classes for label index 0.")

    feature_columns = parse_feature_columns(args.feature_columns)
    flat_features, df, columns_used = load_csv_features(
        csv_path=csv_path,
        delimiter=args.delimiter,
        sensor=args.sensor,
        feature_count=args.feature_count,
        feature_columns=feature_columns,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model, classifier_model, pretrain_cfg = build_models(
        device=device,
        label_num=label_num,
        classifier_version=args.classifier_version,
        pretrain_model_path=Path(args.pretrain_model),
        classifier_model_path=Path(args.classifier_model),
    )

    window_size = args.window_size if args.window_size is not None else pretrain_cfg.seq_len
    stride = args.stride if args.stride is not None else window_size

    data = window_features(flat_features, window_size=window_size, stride=stride)
    data = normalize_sequence_data(data, args.feature_count)
    predictions = predict(data, pretrain_model, classifier_model, args.batch_size, device)

    labels = load_labels(Path(args.label_path)) if args.label_path else None

    print("=== CSV Inference Summary ===")
    print(f"CSV file: {csv_path}")
    print(f"CSV rows: {len(df)}")
    print(f"Columns used ({len(columns_used)}): {columns_used}")
    print(f"Window size: {window_size}, stride: {stride}, windows: {data.shape[0]}")
    print(f"Input tensor shape: {data.shape}")
    print(f"Dataset config: {args.dataset}_{args.dataset_version}")
    print(f"Classes: {label_num}")
    if label_names:
        print(f"Label names: {label_names}")
    print(f"Device: {device}")

    print("\nFirst 20 predictions:")
    for i, pred in enumerate(predictions[:20]):
        name = label_names[pred] if label_names and 0 <= pred < len(label_names) else str(pred)
        print(f"{i:04d}: {pred} ({name})")

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        name = label_names[cls] if label_names and 0 <= cls < len(label_names) else str(cls)
        print(f"{cls}: {count} ({name})")

    if labels is not None:
        if labels.shape[0] != predictions.shape[0]:
            print(
                f"\nSkipped accuracy: label size ({labels.shape[0]}) "
                f"does not match prediction size ({predictions.shape[0]})."
            )
        else:
            accuracy = float(np.mean(predictions == labels))
            print(f"\nAccuracy against {args.label_path}: {accuracy:.4f}")

    if args.output:
        np.save(args.output, predictions)
        print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
