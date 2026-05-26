# `ssl_compare/` — three-way in-domain SSL comparison

Settles the question raised by the LIMU-BERT-X paper (its own Shoaib→Yangzhou
transfer *degraded*, line 163): on Camargo, does reusing the phone-HAR foundation
model help, hurt, or not matter — versus pretraining in-domain from scratch?

## The three recipes

All share the same Camargo windows, mask config, rotation+noise augmentation, and
per-seed train/vali/test split. They differ **only** in (init, lr, epochs):

| mode | init | lr | epochs | meaning |
|------|------|----|--------|---------|
| `scratch` | random | 1e-3 | 1200 | from-scratch in-domain SSL |
| `warmstart` | foundation ckpt | 1e-3 | 1200 | warm-start in-domain SSL |
| `dapt` | foundation ckpt | 1e-4 | 300 | naive DAPT (the old gentle recipe) |

`scratch` vs `warmstart` isolates "does the foundation init help at full lr".
`warmstart` vs `dapt` isolates "full relearn vs gentle adapt".

## Files

- `pretrain_ssl.py` — unified pretraining; `--mode {scratch,warmstart,dapt}`.
  Emits one per-seed ckpt `saved/pretrain_base_<ds>_<ver>/<mode>_seed<seed>.pt`
  (per-seed so the benchmark's held-out 10% test stays unseen — same design as
  `pretrain_dapt.py`).
- `run_ssl_compare.py` — orchestrator: pretrains all modes, then evaluates each
  via `benchmark_results/bench_eval.py` (default `bert_separated` = frozen feature
  extractor + GRU head) with the seed-matching checkpoint, and writes
  `results/ssl_compare_summary.csv` + a mean±std F1 table. Adds a supervised R-GRU
  yardstick row by default.

## Run (from repo root)

```bash
# everything: pretrain 3 modes + eval + table
python ssl_compare/run_ssl_compare.py --gpu 0

# just preview the commands
python ssl_compare/run_ssl_compare.py --dry

# reuse existing checkpoints, only re-evaluate
python ssl_compare/run_ssl_compare.py --skip_pretrain --gpu 0

# one mode's pretraining by hand
python ssl_compare/pretrain_ssl.py v3 camargo 10_20_dense_8cls --mode warmstart \
    -f saved/pretrain_base_camargo_10_20_dense_8cls/limu_bert_x \
    --seeds 3431,42,2026 -g 0
```

Edit the `CONFIG` block at the top of `run_ssl_compare.py` to change
dataset/seeds/label_rates, per-mode lr/epochs, `EVAL_MODE` (`bert_separated` vs
joint `bert` finetune), or to drop the supervised reference.

## Fairness notes

- Augmentation (`AUGMENT=1`) is held constant across all three SSL modes; set to 0
  to reproduce the original clean DAPT.
- Downstream training uses the shared `Recipe` via `bench_eval.py`, identical to the
  main benchmark, so supervised and BERT paths stay comparable.
- To equalize the training budget exactly, set the same `epochs` for all modes in
  `SSL_MODES` (default gives `dapt` fewer, matching its "adapt not relearn" intent).
