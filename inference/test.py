import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import LIMUBertModel4Pretrain, fetch_classifier
from utils import Preprocess4Normalization


DEFAULT_DATA_PATH = Path("dataprep/data_20_120.npy")
DEFAULT_LABEL_PATH = Path("dataprep/label_20_120.npy")
DEFAULT_PRETRAIN_MODEL = Path("saved/pretrain_base_motion_20_120/motion.pt")
DEFAULT_CLASSIFIER_MODEL = Path("saved/classifier_base_gru_motion_20_120/motion.pt")


def resolve_label_path(data_path: Path, explicit_label_path: str | None) -> Path | None:
    if explicit_label_path:
        return Path(explicit_label_path)

    candidate = data_path.with_name(data_path.name.replace("data_", "label_"))
    if candidate.exists():
        return candidate

    if DEFAULT_LABEL_PATH.exists() and data_path.name == DEFAULT_DATA_PATH.name:
        return DEFAULT_LABEL_PATH

    return None


def load_sequence_data(input_path: Path, feature_count: int) -> np.ndarray:
    data = np.load(input_path).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped like (N, W, F), got {data.shape}.")
    if data.shape[-1] < feature_count:
        raise ValueError(
            f"Input has only {data.shape[-1]} features, but {feature_count} are required."
        )
    if data.shape[-1] > feature_count:
        data = data[..., :feature_count]
    return data


def normalize_sequence_data(data: np.ndarray, feature_count: int) -> np.ndarray:
    normalizer = Preprocess4Normalization(feature_count)
    return np.stack([normalizer(sample) for sample in data], axis=0).astype(np.float32)


def load_labels(label_path: Path | None, label_index: int) -> np.ndarray | None:
    if label_path is None:
        return None
    labels = np.load(label_path).astype(np.float32)
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D label array, got {labels.shape}.")
    return labels[:, 0, label_index].astype(np.int64)


def build_models(device: torch.device, classifier_version: str = "v2"):
    pretrain_cfg = load_model_config("pretrain_base", "base", "v1")
    classifier_cfg = load_model_config("classifier_base_gru", "gru", classifier_version)
    if pretrain_cfg is None:
        raise ValueError("Unable to load pretrain model config for base_v1.")
    if classifier_cfg is None:
        raise ValueError(f"Unable to load classifier model config for gru_{classifier_version}.")

    pretrain_model = LIMUBertModel4Pretrain(pretrain_cfg, output_embed=True).to(device)
    classifier_model = fetch_classifier("gru", classifier_cfg, input=pretrain_cfg.hidden, output=6).to(device)

    pretrain_state = torch.load(DEFAULT_PRETRAIN_MODEL, map_location=device)
    classifier_state = torch.load(DEFAULT_CLASSIFIER_MODEL, map_location=device)
    pretrain_model.load_state_dict(pretrain_state)
    classifier_model.load_state_dict(classifier_state)

    pretrain_model.eval()
    classifier_model.eval()
    return pretrain_model, classifier_model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Motion activity labels with the saved classifier.")
    parser.add_argument("input_path", nargs="?", default=str(DEFAULT_DATA_PATH), help="Path to data_20_120.npy")
    parser.add_argument("--label-path", type=str, default=None, help="Optional path to label_20_120.npy")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--feature-count", type=int, default=6, help="Number of raw sensor features to use.")
    parser.add_argument("--classifier-version", type=str, default="v2", choices=["v1", "v2"], help="Classifier config version.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save predictions as .npy")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    label_path = resolve_label_path(input_path, args.label_path)
    dataset_cfg = load_dataset_stats("motion", "20_120")
    if dataset_cfg is None:
        raise ValueError("Unable to load motion_20_120 dataset configuration.")

    label_names, label_num = load_dataset_label_names(dataset_cfg, 0)
    print(f"Dataset: motion_20_120, Features: {args.feature_count}, Classes: {label_num}")
    # Print the first 10 entries from raw data for a quick sanity check.
    raw_loaded = np.load(input_path)
    print("First 10 lines of raw data:")
    if isinstance(raw_loaded, np.lib.npyio.NpzFile):
        first_key = raw_loaded.files[0]
        print(f"Loaded npz key: {first_key}")
        print(raw_loaded[first_key][:10])
        raw_loaded.close()
    else:
        print(raw_loaded[:10])
    data = load_sequence_data(input_path, args.feature_count)
    print(f"Loaded data from {input_path} with shape {data.shape}.")
    data = normalize_sequence_data(data, args.feature_count)
    labels = load_labels(label_path, 0)

    device = torch.device("cuda")
    pretrain_model, classifier_model = build_models(device, classifier_version=args.classifier_version)
    predictions = predict(data, pretrain_model, classifier_model, args.batch_size, device)

    print("=== Prediction Summary ===")
    print(f"Input file: {input_path}")
    print(f"Data shape used for inference: {data.shape}")
    print(f"Device: {device}")
    print(f"Classes: {label_num}")
    if label_names:
        print(f"Label names: {label_names}")

    print("\nFirst 20 predictions:")
    preview = predictions[:20]
    for i, pred in enumerate(preview):
        name = label_names[pred] if label_names and 0 <= pred < len(label_names) else str(pred)
        print(f"{i:04d}: {pred} ({name})")

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        name = label_names[cls] if label_names and 0 <= cls < len(label_names) else str(cls)
        print(f"{cls}: {count} ({name})")

    if labels is not None:
        accuracy = float(np.mean(predictions == labels))
        print(f"\nAccuracy against {label_path}: {accuracy:.4f}")

    if args.output:
        np.save(args.output, predictions)
        print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
