# Benchmark Results

Self-contained benchmarking harness for comparing supervised baselines (DCNN /
DeepSense / R-GRU) against the LIMU-BERT-X foundation model on Camargo.

## Layout

```
benchmark_results/
├── bench_eval.py        # single-run worker, called per (method, label_rate, seed)
├── run_benchmark.py     # orchestrator; loops the config matrix, tees logs + writes CSV
├── plot_benchmark.py    # reads results/summary.csv → plots/*.png
├── logs/                # one .log per run (stdout + stderr)
├── results/             # one .json per run + summary.csv
└── plots/               # label-rate curves + confusion matrices
```

## Run

From the repo root (`D:\01_Code\LIMU-BERT-Public`):

```powershell
python benchmark_results/run_benchmark.py --gpu 0
python benchmark_results/plot_benchmark.py
```

Optional flags for the runner:

- `--dry` print commands without executing.
- `--only DCNN,LIMU` only run rows whose tag matches any substring.
- `--label_rates 0.01,0.1,1.0` override the default label-rate sweep.
- `--seeds 3431,42` override the seeds.

## Editing the config matrix

Open `run_benchmark.py` and edit the constants at the top:

- `DATASET`, `DATASET_VERSION`, `MODEL_VERSION` — must match what exists in
  `dataset/<DATASET>/data_<DATASET_VERSION>.npy` and the JSON model configs.
- `LIMU_BERTX_CKPT` — path (relative to repo root) of the foundation-model
  checkpoint. The default points at
  `saved/pretrain_base_camargo_10_20/limu_bert_x.pt`.
- `LABEL_INDEX` — which label column to predict (0 = activity for camargo).
  **Don't set to -1**: in some dataset configs that matches a
  `_label_index: -1` sentinel and yields `label_num=0`, which crashes CE with
  a CUDA `t < n_classes` assert.
- `LABEL_RATES`, `SEEDS` — sweep parameters.
- All runs use the single `Recipe.default()` defined in `recipe.py`: sqrt
  class-weighted CE (matches classifier.py), no rare-class filter, no LR
  scheduler, early stopping (patience=10), and `lr_scale=0.1`. Applied
  identically to supervised, bert-joint, and bert-separated paths so the
  comparison is apples-to-apples.
- `RUNS` — list of model configurations to evaluate. Each entry is a dict;
  `mode="supervised"` calls `benchmark.py:classify_benchmark`, `mode="bert"`
  calls `classifier_bert.py:bert_classify` (which loads the pretrained
  LIMU-BERT-X weights), `mode="bert_separated"` runs the
  `embedding.py + classifier.py` two-stage path. Optional per-run key:
  `label_index`.

## Outputs

- `results/<run_id>.json` — per-run metrics + confusion matrix.
- `results/summary.csv` — appended every run; one row per `(tag, label_rate, seed)`.
- `logs/<run_id>.log` — full stdout for each run (training curves, etc.).
- `plots/labelrate_vs_f1.png` — main story plot (one curve per tag, seed mean ± std).
- `plots/labelrate_vs_acc.png` — same but accuracy.
- `plots/confusion_*.png` — per-run normalized confusion matrices.

## Notes

- The runner calls `benchmark.py:classify_benchmark`,
  `classifier_bert.py:bert_classify`, and `classifier.py:classify_embeddings`
  via subprocess. All three accept a `recipe` argument (see `recipe.py`);
  `bench_eval.py` builds one `Recipe.default()` per run and passes it to
  every path so the comparison is fair.
- Each run writes a temporary `config/bench_tmp_*.json` so the seed can be
  overridden cleanly; the file is removed when the run finishes.
- `bench_eval.py` sets `classifier_bert.method` before calling `bert_classify`
  because the original function references `method` as a module-level free
  variable rather than a parameter.
- Failed runs are recorded in `summary.csv` with `status=failed(...)` so they
  do not pollute plots, which only use rows with `status=ok`.
- The CSV schema has a `recipe` column that always reads `"default"` after
  the unification; pre-unification rows ("filtered" / "vanilla" /
  "classifier_py_default") are kept so old sweeps can be filtered out.
