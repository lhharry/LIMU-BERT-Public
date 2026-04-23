# LIMU-BERT-Public — Project Folder Description

> Auto-generated snapshot for quick context loading. Last updated: 2026-04-22.

## What this project is

LIMU-BERT is a self-supervised representation learning model for IMU (accelerometer / gyroscope) sensor data, built on the BERT principle. It learns generalised temporal features from *unlabeled* IMU data, then fine-tunes lightweight task-specific classifiers (GRU, LSTM, CNN, Attention) on top. The codebase targets Human Activity Recognition (HAR) tasks.

Paper: [LIMU-BERT @ ACM SenSys 2021](https://dl.acm.org/doi/10.1145/3485730.3485937)

---

## Two-Phase Framework

1. **Self-supervised pre-training** — `pretrain.py` trains `LIMUBertModel4Pretrain` on unlabeled IMU windows using masked-sensor prediction (span masking, like BERT's MLM).
2. **Supervised fine-tuning** — `embedding.py` extracts BERT embeddings → `classifier.py` trains a GRU/LSTM/CNN head on labeled data.

---

## Top-Level Files

| File | Role |
|---|---|
| `models.py` | All model architectures (see below) |
| `config.py` | NamedTuple configs + JSON loaders for model/train/mask/dataset |
| `utils.py` | Data splitting, masking pipelines, Dataset classes, CLI arg parsing |
| `train.py` | Training loop helpers |
| `pretrain.py` | Entry point: pre-train LIMU-BERT |
| `embedding.py` | Entry point: generate and save BERT embeddings → `embed/` |
| `classifier.py` | Entry point: train GRU classifier on pre-computed embeddings |
| `classifier_bert.py` | Entry point: train BERT+GRU jointly (fine-tune end-to-end) |
| `benchmark.py` | Entry point: train baseline models (DCNN, DeepSense, R-GRU) |
| `statistic.py` | Evaluation helpers (accuracy, F1, etc.) |
| `plot.py` | Plotting helpers for IMU data and embeddings |
| `pretrain.sh` | Shell script to batch-run pre-training |
| `requirements.txt` | Python dependencies (PyTorch 1.5–1.7, NumPy, SciPy, etc.) |

---

## Models (`models.py`)

### LIMU-BERT Backbone
- `Embeddings` — linear projection (feature → hidden) + learned positional embedding + LayerNorm
- `MultiHeadedSelfAttention` — multi-head dot-product attention with parameter-sharing across layers
- `PositionWiseFeedForward` — two-layer FFN with GELU activation
- `Transformer` — stacks the above (with parameter-sharing across `n_layers`)
- `LIMUBertModel4Pretrain` — Transformer encoder + masked-position decoder head; set `output_embed=True` to get embeddings instead of logits

### Task-Specific Classifiers
- `ClassifierGRU` — configurable multi-layer GRU + linear head (default downstream model)
- `ClassifierLSTM` — same structure with LSTM
- `ClassifierCNN1D` / `ClassifierCNN2D` — 1-D / 2-D convolution + pooling + linear
- `ClassifierAttn` — positional embedding + multi-head self-attention + linear
- `BERTClassifier` — wraps Transformer + any classifier for joint training
- `fetch_classifier(method, cfg, ...)` — factory function to select classifier by name

### Baselines
- `BenchmarkDCNN` — two-layer 2-D CNN + linear (DCNN baseline)
- `BenchmarkDeepSense` — per-sensor CNN block + cross-sensor CNN + linear (DeepSense baseline)
- `BenchmarkTPNPretrain` / `BenchmarkTPNClassifier` — TPN self-supervised baseline

---

## Configuration (`config/`)

| File | Purpose |
|---|---|
| `limu_bert.json` | BERT model configs keyed `base_v1`, `base_v2`, … (`hidden`, `hidden_ff`, `n_layers`, `n_heads`, `seq_len`, `feature_num`) |
| `classifier.json` | Classifier configs keyed `gru_v1`, `gru_v2`, `dcnn_v1`, … |
| `pretrain.json` | Pre-training hyperparameters (`lr`, `batch_size`, `n_epochs`, `warmup`, etc.) |
| `train.json` | Fine-tuning / classifier training hyperparameters |
| `bert_classifier_train.json` | Hyperparameters for joint BERT+classifier training |
| `mask.json` | Masking strategy (`mask_ratio`, `mask_alpha`, `max_gram`, `mask_prob`, `replace_prob`) |

---

## Datasets (`dataset/`)

Four standard HAR datasets, each preprocessed into `data_20_120.npy` (shape `N×120×F`) and `label_20_120.npy` (shape `N×120×L`).

| Dataset | Folder | Script | Labels |
|---|---|---|---|
| HHAR | `dataset/hhar/` | `hhar.py` | activity, user, device, model |
| UCI | `dataset/uci/` | `uci.py` | activity, user |
| MotionSense | `dataset/motion/` | `motion.py` | activity, user |
| Shoaib | `dataset/shoaib/` | `shoaib.py` | activity, position |

`data_config.json` — per-dataset metadata: sampling rate, dimension, label sizes and names, label indices.

Naming convention: `data_<sr>_<window>.npy` where `sr=20` Hz and `window=120` samples (6 seconds).

---

## Saved Models (`saved/`)

Pre-trained `.pt` files, one per dataset:

```
saved/
  pretrain_base_{dataset}_20_120/{dataset}.pt     ← pre-trained LIMU-BERT
  classifier_base_gru_{dataset}_20_120/{dataset}.pt ← fine-tuned GRU classifier
```

Datasets: `hhar`, `motion`, `shoaib`, `uci`.

---

## Data Preparation (`dataprep/`)

Scripts for custom/external datasets (not the four standard ones):

| File | Purpose |
|---|---|
| `build_training_data.py` | Merges IMU CSVs + condition/label CSVs for the AY dataset (`D:\DATA\OpenSource\AY_Data\ABxx\`) into `training_data/` per subject |
| `mat2csv.py` / `mat2npy.py` | Convert `.mat` files to CSV / `.npy` |
| `preprocess_ay.py` | Preprocessing pipeline for the AY (Camargo) dataset |
| `speed_to_label.py` | Maps treadmill speed values to discrete activity labels |
| `count_labels.py` | Counts label distribution in a `.npy` file |
| `folder.py` | File system utilities |
| `camargo_mat2csv.m` | MATLAB script to convert Camargo dataset `.mat` → CSV |
| `dataparse_stair.m` / `dataparse_treadmill.m` | MATLAB parsers for stair/treadmill data |
| `stair/` | Sample stair data (`.mat`, `.csv`, `.npy`) |
| `walking/` | Sample treadmill/walking data (`.mat`, `.csv`) |

---

## Inference (`inference/`)

Scripts for running inference on new/real-world data:

| File | Purpose |
|---|---|
| `test.py` | **Main inference entry point**: loads `motion_20_120` pretrained LIMU-BERT + GRU classifier, runs batch prediction on any `data_20_120.npy`, prints per-class distribution and optional accuracy |
| `test_csv.py` | CSV-based inference variant |
| `log.py` | Logging utilities for inference runs |
| `plot_training_data.py` | Visualise training `.npy` data as time-series plots |
| `read_npy.py` | Inspect / print `.npy` file contents |
| `imu_log_300s.csv` | 300-second raw IMU log (real capture) |
| `imu_log_300s_corrected.csv` | Corrected version of the above |

`test.py` default model paths:
- Pretrain: `saved/pretrain_base_motion_20_120/motion.pt`
- Classifier: `saved/classifier_base_gru_motion_20_120/motion.pt`
- Output: 6-class activity prediction

---

## Molinaro Dataset (`molinaro/`)

| File | Purpose |
|---|---|
| `preprocess_molinaro.py` | Preprocessing for the Molinaro dataset |
| `compute_alignment.py` | Temporal alignment between Molinaro sensor streams |

---

## Key Data Shapes

| Array | Shape | Notes |
|---|---|---|
| Raw data | `(N, W, F)` | N samples, W=120 window, F=6 (acc+gyro) or 9 (+mag) |
| Labels | `(N, W, L)` | L label types per timestep |
| Embeddings | `(N, W, hidden)` | hidden=72 for base_v1 |

---

## CLI Usage Pattern (shared by all entry points)

```
python <script>.py <model_version> <dataset> <dataset_version> [options]

# model_version: v1, v2, v1_v2 (bert+classifier)
# dataset: hhar | motion | uci | shoaib
# dataset_version: 10_100 | 20_120

# Examples:
python pretrain.py v1 uci 20_120 -s limu_v1
python embedding.py v1 uci 20_120 -f limu_v1
python classifier.py v2 uci 20_120 -f limu_v1 -s limu_gru_v1 -l 0
python benchmark.py v1 uci 20_120 -s dcnn_v1 -l 0
python inference/test.py [data_path] [--label-path ...] [--batch-size 128]
```
