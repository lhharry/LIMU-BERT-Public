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
- `--recipe {vanilla,filtered}` override `DEFAULT_RECIPE` for every run in the
  invocation. The chosen recipe is applied identically to **both** the
  supervised and the BERT path so the comparison is apples-to-apples.

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
- `DEFAULT_RECIPE` — `"filtered"` (drop classes with <5 train samples,
  class-weighted CE, cosine LR, early stop, lr×0.1) or `"vanilla"` (none of
  the above). Applied identically to both paths. Per-run override via
  `run_cfg["recipe"]`. The 5-sample threshold auto-falls-back to "keep every
  class with >=1 sample" when no class would survive — necessary so the same
  recipe works at very low `label_rate` (e.g. 0.01). Vali/test rows for
  classes absent from the filtered training set are dropped, so metrics
  reflect only the surviving label space.
- `RUNS` — list of model configurations to evaluate. Each entry is a dict;
  `mode="supervised"` calls `benchmark.py:classify_benchmark`, `mode="bert"`
  calls `classifier_bert.py:bert_classify` (which loads the pretrained
  LIMU-BERT-X weights). Optional per-run keys: `recipe`, `label_index`.

## Outputs

- `results/<run_id>.json` — per-run metrics + confusion matrix.
- `results/summary.csv` — appended every run; one row per `(tag, label_rate, seed)`.
- `logs/<run_id>.log` — full stdout for each run (training curves, etc.).
- `plots/labelrate_vs_f1.png` — main story plot (one curve per tag, seed mean ± std).
- `plots/labelrate_vs_acc.png` — same but accuracy.
- `plots/confusion_*.png` — per-run normalized confusion matrices.

## Notes

- The runner calls the repo's `benchmark.py:classify_benchmark` and
  `classifier_bert.py:bert_classify` via subprocess. Both functions accept a
  shared `recipe` argument (see `recipe.py`); `bench_eval.py` builds one
  `Recipe` per run and passes it to both paths so the comparison is fair.
- The recipe used to be hardcoded inside `bert_classify` (rare-class drop,
  class-weighted CE, cosine LR, early stop, lr×0.1) and absent from
  `classify_benchmark`, which made the BERT vs supervised numbers
  incomparable. That is now `Recipe.filtered()` and applies to both paths.
- Each run writes a temporary `config/bench_tmp_*.json` so the seed can be
  overridden cleanly; the file is removed when the run finishes.
- `bench_eval.py` sets `classifier_bert.method` before calling `bert_classify`
  because the original function references `method` as a module-level free
  variable rather than a parameter.
- Failed runs are recorded in `summary.csv` with `status=failed(...)` so they
  do not pollute plots, which only use rows with `status=ok`.
- The CSV schema gained a `recipe` column; delete or rotate any pre-existing
  `results/summary.csv` before the next sweep so the header matches.
