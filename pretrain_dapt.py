#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : Domain-adaptive continued pretraining (DAPT) for LIMU-BERT.
"""
Continue the self-supervised MLM/reconstruction pretraining of an EXISTING
(e.g. foreign WANDS-HKUST) LIMU-BERT checkpoint on a target dataset's UNLABELED
training-split windows, producing ONE adapted checkpoint per seed.

Why per-seed: the downstream benchmark (benchmark_results/bench_eval.py) re-splits
the data with its --seed via utils.partition_and_reshape (set_seeds(seed) then
np.random.shuffle). Pretraining MUST use the same seed so the held-out 10% test
split is identical and never seen during pretraining -- otherwise the adapted
model leaks test windows. So we emit one ckpt per benchmark seed and pair them
in run_benchmark.py (bert run with --seed S loads ..._seed{S}.pt).

The 10% test split is never returned by prepare_pretrain_dataset, so it is fully
held out; only the 80% train (optimized) + 10% vali (best-model selection) are used.

Usage (run on the server, alongside pretrain.py):
  python pretrain_dapt.py v3 camargo 10_20_dense_8cls \
      -f saved/bertx/limu_bert_x \
      --seeds 3431,42,2026 --dapt_epochs 200 --dapt_lr 1e-4 -g 0
  python pretrain_dapt.py v3 merged 10_20_9cls_align \
      -f saved/bertx/limu_bert_x \
      --seeds 3431 --dapt_epochs 3200 --dapt_lr 1e-4 -g 0 --out_name limu_bert_x_align_dapt_1e-4_3200

Output: saved/pretrain_base_<dataset>_<version>/<out_name>_seed<seed>.pt
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train
from config import (create_io_config, load_model_config, load_dataset_stats,
                    TrainConfig, MaskConfig)
from models import LIMUBertModel4Pretrain
from utils import (set_seeds, get_device, LIBERTDataset4Pretrain,
                   Preprocess4Normalization, Preprocess4Mask,
                   prepare_pretrain_dataset)

TARGET = "pretrain_base"   # -> save dir saved/pretrain_base_<dataset>_<version>/
PREFIX = "base"            # -> model config key base_<model_version> in config/limu_bert.json


def parse_args():
    p = argparse.ArgumentParser(description="LIMU-BERT domain-adaptive continued pretraining")
    # Positional args mirror handle_argv (pretrain.py) so configs resolve identically.
    p.add_argument("model_version", type=str, help="BERT config version, e.g. v3 (-> base_v3)")
    p.add_argument("dataset", type=str,
                   choices=["hhar", "motion", "uci", "shoaib", "camargo", "merged"])
    p.add_argument("dataset_version", type=str,
                   choices=["10_100", "20_120", "10_20", "10_60", "10_20_dense",
                            "10_20_dense_8cls", "10_20_9cls_align"])
    p.add_argument("-f", "--model_file", type=str, required=True,
                   help="Starting checkpoint to continue from, e.g. saved/bertx/limu_bert_x "
                        "(a trailing .pt is optional).")
    p.add_argument("-g", "--gpu", type=str, default=None, help="Specific GPU id")
    p.add_argument("-t", "--train_cfg", type=str, default="./config/pretrain.json",
                   help="Base training config (seed/lr/epochs overridden by flags below).")
    p.add_argument("-a", "--mask_cfg", type=str, default="./config/mask.json")
    p.add_argument("--seeds", type=str, default="3431,42,2026",
                   help="Comma-separated seeds; one ckpt per seed, aligned with benchmark seeds.")
    p.add_argument("--training_rate", type=float, default=0.8,
                   help="Must match benchmark TRAINING_RATE so the test split aligns.")
    p.add_argument("--dapt_epochs", type=int, default=200,
                   help="Continued-pretraining epochs (fewer than from-scratch; we adapt, not learn).")
    p.add_argument("--dapt_lr", type=float, default=1e-4,
                   help="Continued-pretraining LR (smaller than from-scratch 1e-3 to avoid "
                        "catastrophic forgetting of the foundation model).")
    p.add_argument("--out_name", type=str, default="limu_bert_x_dapt",
                   help="Output ckpt name stem; final file is <out_name>_seed<seed>.pt.")
    return p.parse_args()


def build_io(args):
    """Resolve model/dataset config + save dir exactly like utils.handle_argv does."""
    model_cfg = load_model_config(TARGET, PREFIX, args.model_version)
    if model_cfg is None:
        sys.exit("Unable to find model config %s_%s in config/limu_bert.json" % (PREFIX, args.model_version))
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        sys.exit("Unable to find dataset config %s_%s" % (args.dataset, args.dataset_version))
    if model_cfg.feature_num > dataset_cfg.dimension:
        sys.exit("model feature_num (%d) > dataset dimension (%d)" % (model_cfg.feature_num, dataset_cfg.dimension))
    args.model_cfg = model_cfg
    args.dataset_cfg = dataset_cfg
    args.save_model = args.out_name  # placeholder; per-seed save_path set in the loop
    args = create_io_config(args, args.dataset, args.dataset_version,
                            pretrain_model=args.model_file, target=TARGET)
    return args


def pretrain_one_seed(args, base_train_cfg, mask_cfg, seed, save_dir, device):
    # TrainConfig is a NamedTuple -> _replace gives an overridden copy.
    train_cfg = base_train_cfg._replace(seed=seed, lr=args.dapt_lr, n_epochs=args.dapt_epochs)

    set_seeds(seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)

    # Same seed + same partition_and_reshape as the classifier path => identical
    # train/vali/test split. The 10% test is NOT returned here, so it is held out.
    data_train, _, data_vali, _ = prepare_pretrain_dataset(data, labels, args.training_rate, seed=seed)

    pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    loader_train = DataLoader(LIBERTDataset4Pretrain(data_train, pipeline=pipeline),
                              shuffle=True, batch_size=train_cfg.batch_size)
    loader_vali = DataLoader(LIBERTDataset4Pretrain(data_vali, pipeline=pipeline),
                             shuffle=False, batch_size=train_cfg.batch_size)

    model = LIMUBertModel4Pretrain(args.model_cfg)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

    out_stem = os.path.join(save_dir, "%s_seed%d" % (args.out_name, seed))
    trainer = train.Trainer(train_cfg, model, optimizer, out_stem, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        return criterion(model(mask_seqs, masked_pos), seqs)

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        return model(mask_seqs, masked_pos), seqs

    def func_evaluate(seqs, predict_seqs):
        return criterion(predict_seqs, seqs).mean().cpu().numpy()

    print("\n=== DAPT seed=%d | train=%d vali=%d | start=%s | lr=%g epochs=%d ==="
          % (seed, data_train.shape[0], data_vali.shape[0], args.pretrain_model,
             train_cfg.lr, train_cfg.n_epochs))
    print("    mask_cfg(%s): ratio=%g alpha=%g max_gram=%g prob=%g replace=%g"
          % (args.mask_cfg, mask_cfg.mask_ratio, mask_cfg.mask_alpha,
             mask_cfg.max_gram, mask_cfg.mask_prob, mask_cfg.replace_prob))
    # model_file = the (stripped) starting ckpt; Trainer.load re-appends '.pt'.
    trainer.pretrain(func_loss, func_forward, func_evaluate, loader_train, loader_vali,
                     model_file=args.pretrain_model)
    saved = out_stem + ".pt"
    print("Saved adapted checkpoint -> %s" % saved)
    return saved


def main():
    args = parse_args()
    args = build_io(args)
    base_train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    device = get_device(args.gpu)
    save_dir = os.path.dirname(args.save_path)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    produced = {}
    for seed in seeds:
        produced[seed] = pretrain_one_seed(args, base_train_cfg, mask_cfg, seed, save_dir, device)

    print("\n=== DAPT done. Pair each benchmark seed with its checkpoint in run_benchmark.py: ===")
    for seed, path in produced.items():
        print("  seed %-6d -> %s" % (seed, path))


if __name__ == "__main__":
    main()
