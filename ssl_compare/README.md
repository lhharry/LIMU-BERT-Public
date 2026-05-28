# `ssl_compare/` — three-way in-domain SSL comparison

Settles the question raised by the LIMU-BERT-X paper (its own Shoaib→Yangzhou
transfer *degraded*, line 163): on Camargo, does reusing the phone-HAR foundation
model help, hurt, or not matter — versus pretraining in-domain from scratch?

## The three pretrain recipes

All share the same Camargo windows, mask config, rotation+noise augmentation, and
per-seed train/vali/test split. They differ **only** in (init, lr, epochs):

| mode | init | lr | epochs | meaning |
|------|------|----|--------|---------|
| `scratch` | random | 1e-3 | 1200 | from-scratch in-domain SSL |
| `warmstart` | foundation ckpt | 1e-3 | 1200 | warm-start in-domain SSL |
| `dapt` | foundation ckpt | 1e-4 | 300 | naive DAPT (the old gentle recipe) |

`scratch` vs `warmstart` isolates "does the foundation init help at full lr".
`warmstart` vs `dapt` isolates "full relearn vs gentle adapt".

## Downstream evaluation matrix

`run_ssl_compare.py` no longer runs a single eval path — `EVAL_MODES` is a list, and
for each SSL mode it emits one row per entry. Default = both rows:

| eval mode | row tag | bench_eval flags | what trains downstream |
|-----------|---------|------------------|------------------------|
| `bert_separated` | `<mode>` | `--mode bert_separated --method gru` | BERT frozen (eval/no_grad), embeddings cached, standalone GRU head |
| `bert` | `<mode>_ft` | `--mode bert --method base_gru --frozen_bert 0` | BERT + GRU head co-trained from the SSL ckpt |

The separated row is the representation-quality probe (matches
`inference/test_csv.py`). The `_ft` row is the joint-finetune number that's
directly comparable to the supervised R-GRU yardstick. Both paths share the same
`Recipe` defaults (warmup_epochs=20, cosine_decay=True, early_stop_patience=15,
sqrt-weighted balanced CE) — see `recipe.py`.

Edit `EVAL_MODES` in `run_ssl_compare.py` to drop a path or add custom ones.

## Files

- `pretrain_ssl.py` — unified pretraining; `--mode {scratch,warmstart,dapt}`.
  Emits one per-seed ckpt `saved/pretrain_base_<ds>_<ver>/<mode>_seed<seed>.pt`
  (per-seed so the benchmark's held-out 10% test stays unseen — same design as
  `pretrain_dapt.py`).
- `run_ssl_compare.py` — orchestrator: pretrains all modes, then evaluates each
  through every entry in `EVAL_MODES`, and writes `results/ssl_compare_summary.csv`
  + a mean±std F1 table. Adds a supervised R-GRU yardstick row by default.
- `plot_tsne_compare.py` — side-by-side t-SNE of BERT embeddings from foundation +
  the 3 SSL recipes (same fixed sample of Camargo windows fed through all 4
  ckpts, TSNE with fixed `random_state`, colored by activity).
- `plot_run_summary.py` — one-figure summary of a single run: pretrain MLM
  loss curves (left) + downstream metric vs label_rate (right). Shares a color
  per tag across both panels, so e.g. `scratch`'s loss curve and F1 line are the
  same color. Reads from `--run_dir` (default: live `ssl_compare/`; point at
  `history/RunN_.../` for an archived snapshot).

## Run (from repo root)

```bash
# everything: pretrain 3 modes + eval (both separated and _ft rows) + table
python ssl_compare/run_ssl_compare.py --gpu 0

# just preview the commands
python ssl_compare/run_ssl_compare.py --dry

# reuse existing checkpoints, only re-evaluate
python ssl_compare/run_ssl_compare.py --skip_pretrain --gpu 0

# add a new EVAL_MODES entry without re-running the rows already in results/
python ssl_compare/run_ssl_compare.py --skip_pretrain --skip_existing_eval --gpu 0

# one mode's pretraining by hand
python ssl_compare/pretrain_ssl.py v3 camargo 10_20_dense_8cls --mode warmstart \
    -f saved/pretrain_base_camargo_10_20_dense_8cls/limu_bert_x \
    --seeds 3431,42,2026 -g 0

# 4-panel t-SNE comparison (foundation + scratch + warmstart + dapt)
python ssl_compare/plot_tsne_compare.py -g 0 \
    --out ssl_compare/history/<run-dir>/tsne_compare.png

# one-figure summary of a run: loss curves + F1 vs label_rate
python ssl_compare/plot_run_summary.py \
    --run_dir ssl_compare/history/<run-dir> \
    --out ssl_compare/history/<run-dir>/run_summary.png
```

Edit the `CONFIG` block at the top of `run_ssl_compare.py` to change
dataset / seeds / label_rates / per-mode lr+epochs / `EVAL_MODES` / or to drop the
supervised reference.

## CLI flags worth knowing

| flag | what it does |
|------|--------------|
| `--skip_pretrain` | reuse existing `<mode>_seed<seed>.pt`, jump straight to eval |
| `--skip_eval` | only run the pretrain phase |
| `--skip_existing_ckpt` | during pretrain, skip seeds whose ckpt already exists |
| `--skip_existing_eval` | during eval, reuse rows whose output JSON already exists (lets you add a new `EVAL_MODES` entry without re-running the completed ones) |
| `--only scratch,dapt` | subset of SSL modes |
| `--label_rates 0.01,0.1` | override the default label-rate sweep |
| `--dry` | print every subprocess command, run nothing |

## Fairness notes

- Augmentation (`AUGMENT=1`) is held constant across all three SSL modes; set to 0
  to reproduce the original clean DAPT.
- Both downstream paths (`bert_separated`, `bert` finetune) share the same `Recipe`
  via `bench_eval.py`, identical to the main benchmark — so the SSL rows, their
  `_ft` counterparts, and the supervised R-GRU yardstick are all directly
  comparable.
- The dead fields in `config/train.json` (`lambda1`, `lambda2`, `warmup`,
  `save_steps`, `total_steps`) are not referenced by any active code path; the
  scheduler / loss are entirely owned by `recipe.py`.
- To equalize the training budget exactly, set the same `epochs` for all modes in
  `SSL_MODES` (default gives `dapt` fewer, matching its "adapt not relearn" intent).

## `history/` snapshots

Completed runs are archived under `history/RunN_<date>_<note>/` with their own
`logs/` + `results/` + (optionally) the per-mode `*_seed<seed>.pt` checkpoint
copies. `results/ssl_compare_summary.csv` is the final table for each run;
`tsne_compare.png` and `run_summary.png` (if present) are the visualizations
generated by the two plot scripts above on that snapshot.
