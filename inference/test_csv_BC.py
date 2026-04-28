from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import BERTClassifier, fetch_classifier
from utils import Preprocess4Normalization


DEFAULT_CSV_PATH = Path("inference/data/walking_300s_3.5kmh_imu_log_20260427_171535.csv")
DEFAULT_FINETUNED_MODEL = Path("saved/bert_classifier_base_gru_camargo_10_20_dense/camargo_bertx_0428dense.pt")

DEBUG_CONFIG = {
    "csv_path": DEFAULT_CSV_PATH,
    "dataset": "camargo",
    "dataset_version": "10_20_dense",
    "delimiter": ",",
    "sensor": "left",
    "feature_columns": None,
    "window_size": 20,
    "stride": 20,
    "feature_count": 6,
    "batch_size": 128,
    "bert_version": "v3",            # NEW
    "classifier_version": "v2",
    "finetuned_model": DEFAULT_FINETUNED_MODEL,   # replaces pretrain_model + classifier_model
    "label_num_override": 9,         # NEW — see note below
    "label_path": None,
    "output": None,

}

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


def build_model(
    device: torch.device,
    label_num: int,
    bert_version: str,
    classifier_version: str,
    finetuned_model_path: Path,
):
    bert_cfg = load_model_config("pretrain_base", "base", bert_version)
    classifier_cfg = load_model_config("classifier_base_gru", "gru", classifier_version)
    if bert_cfg is None:
        raise ValueError(f"Unable to load bert config base_{bert_version}")
    if classifier_cfg is None:
        raise ValueError(f"Unable to load classifier config gru_{classifier_version}")

    inner_classifier = fetch_classifier(
        "gru", classifier_cfg, input=bert_cfg.hidden, output=label_num
    )
    model = BERTClassifier(
        bert_cfg, classifier=inner_classifier, frozen_bert=False
    ).to(device)

    if not finetuned_model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {finetuned_model_path}")

    state = torch.load(finetuned_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, bert_cfg


def predict(data: np.ndarray, model, batch_size: int, device: torch.device) -> np.ndarray:
    data_loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            logits = model(batch, False)   # second arg = training flag
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
    cfg = DEBUG_CONFIG
    csv_path = Path(cfg["csv_path"])
    dataset_cfg = load_dataset_stats(cfg["dataset"], cfg["dataset_version"])
    if dataset_cfg is None:
        raise ValueError(f"Unable to load dataset config: {cfg['dataset']}_{cfg['dataset_version']}")

    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    if label_num <= 0:
        raise ValueError("Unable to resolve number of classes for label index 0.")

    feature_columns = parse_feature_columns(cfg["feature_columns"])
    flat_features, df, columns_used = load_csv_features(
        csv_path=csv_path,
        delimiter=cfg["delimiter"],
        sensor=cfg["sensor"],
        feature_count=cfg["feature_count"],
        feature_columns=feature_columns,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use override if classifier was trained with filtered classes
    effective_label_num = cfg.get("label_num_override") or label_num

    model, bert_cfg = build_model(
        device=device,
        label_num=effective_label_num,
        bert_version=cfg["bert_version"],
        classifier_version=cfg["classifier_version"],
        finetuned_model_path=Path(cfg["finetuned_model"]),
    )
    window_size = cfg["window_size"] if cfg["window_size"] is not None else bert_cfg.seq_len
    stride = cfg["stride"] if cfg["stride"] is not None else window_size

    data = window_features(flat_features, window_size=window_size, stride=stride)
    data = normalize_sequence_data(data, cfg["feature_count"])
    predictions = predict(data, model, cfg["batch_size"], device)

    labels = load_labels(Path(cfg["label_path"])) if cfg["label_path"] else None

    print("=== CSV Inference Summary ===")
    print(f"CSV file: {csv_path}")
    print(f"CSV rows: {len(df)}")
    print(f"Columns used ({len(columns_used)}): {columns_used}")
    print(f"Window size: {window_size}, stride: {stride}, windows: {data.shape[0]}")
    print(f"Input tensor shape: {data.shape}")
    print(f"Dataset config: {cfg['dataset']}_{cfg['dataset_version']}")
    print(f"Classes (dataset original): {label_num}")
    print(f"Classes (model output): {effective_label_num}")
    if label_names:
        print(f"Label names: {label_names}")
    print(f"Device: {device}")

    print("\nFirst 50 predictions:")
    for i, pred in enumerate(predictions[:50]):
        name = label_names[pred] if label_names and 0 <= pred < len(label_names) else str(pred)
        print(f"{i:04d}: {pred} ({name})")

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        name = label_names[cls] if label_names and 0 <= cls < len(label_names) else str(cls)
        print(f"class={cls} ({name}): {count}")

    if labels is not None:
        if labels.shape[0] != predictions.shape[0]:
            print(
                f"\nSkipped accuracy: label size ({labels.shape[0]}) "
                f"does not match prediction size ({predictions.shape[0]})."
            )
        else:
            accuracy = float(np.mean(predictions == labels))
            print(f"\nAccuracy against {cfg['label_path']}: {accuracy:.4f}")

    if cfg["output"]:
        np.save(cfg["output"], predictions)
        print(f"Saved predictions to: {cfg['output']}")


if __name__ == "__main__":
    main()
