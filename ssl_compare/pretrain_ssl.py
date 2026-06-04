#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : Unified in-domain SSL pretraining for the 3-way comparison.
"""
One script, three modes -- they differ ONLY in (initialization, lr, epochs):

  scratch   : random init        + full lr (1e-3) + full epochs   -> from-scratch in-domain SSL
  warmstart : foundation-model   + full lr (1e-3) + full epochs   -> warm-start in-domain SSL
  dapt      : foundation-model   + gentle lr (1e-4) + few epochs   -> naive DAPT (the old recipe)

Everything else is held constant across modes so the comparison is fair:
the same Camargo windows, the same mask config, the same rotation+noise
augmentation (toggle with --augment), the same per-seed train/vali/test split.

Per-seed checkpoints (like pretrain_dapt.py): the downstream benchmark re-splits
with its --seed, so we pretrain with the SAME seed to keep the held-out 10% test
split identical and unseen. One ckpt per seed:
  saved/pretrain_base_<dataset>_<version>/<mode>_seed<seed>.pt

Usage (run from repo root, alongside pretrain.py):
  python ssl_compare/pretrain_ssl.py v3 camargo 10_20_dense_8cls --mode warmstart \
      -f saved/pretrain_base_camargo_10_20_dense_8cls/limu_bert_x \
      --seeds 3431,42,2026 -g 0
  python ssl_compare/pretrain_ssl.py v3 camargo 10_20_dense_8cls --mode scratch \
      --seeds 3431,42,2026 -g 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make repo root importable when this file lives in ssl_compare/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import train
from config import (create_io_config, load_model_config, load_dataset_stats,
                    TrainConfig, MaskConfig)
from models import LIMUBertModel4Pretrain
from utils import (set_seeds, get_device, LIBERTDataset4Pretrain,
                   Preprocess4Normalization, Preprocess4Mask, Preprocess4Augment,
                   prepare_pretrain_dataset)

TARGET = "pretrain_base"   # -> save dir saved/pretrain_base_<dataset>_<version>/
PREFIX = "base"            # -> model config key base_<model_version> in config/limu_bert.json

# Per-mode defaults. CLI flags --lr / --epochs override these when set.
MODE_DEFAULTS = {
    "scratch":   {"use_init": False, "lr": 1e-3, "epochs": 1200},
    "warmstart": {"use_init": True,  "lr": 1e-3, "epochs": 1200},
    "dapt":      {"use_init": True,  "lr": 1e-4, "epochs": 300},
}


def parse_args():
    p = argparse.ArgumentParser(description="Unified in-domain SSL pretraining (scratch/warmstart/dapt)")
    # Positional args mirror handle_argv (pretrain.py) so configs resolve identically.
    p.add_argument("model_version", type=str, help="BERT config version, e.g. v3 (-> base_v3)")
    p.add_argument("dataset", type=str,
                   choices=["hhar", "motion", "uci", "shoaib", "camargo",
                            "molinaro", "scherpereel", "scherpereel_exo"])
    p.add_argument("dataset_version", type=str,
                   choices=["10_100", "20_120", "10_20", "10_60", "10_20_dense",
                            "10_20_dense_8cls", "10_20_both"])
    p.add_argument("--mode", required=True, choices=list(MODE_DEFAULTS.keys()))
    p.add_argument("-f", "--model_file", type=str, default=None,
                   help="Starting checkpoint for warmstart/dapt, e.g. "
                        "saved/pretrain_base_.../limu_bert_x (trailing .pt optional). "
                        "Ignored for --mode scratch.")
    p.add_argument("-g", "--gpu", type=str, default=None, help="Specific GPU id")
    p.add_argument("-t", "--train_cfg", type=str, default="./config/pretrain.json",
                   help="Base training config (seed/lr/epochs overridden by flags/mode).")
    p.add_argument("-a", "--mask_cfg", type=str, default="./config/mask.json")
    p.add_argument("--seeds", type=str, default="3431,42,2026",
                   help="Comma-separated seeds; one ckpt per seed, aligned with benchmark seeds.")
    p.add_argument("--training_rate", type=float, default=0.8,
                   help="Must match benchmark TRAINING_RATE so the test split aligns.")
    p.add_argument("--lr", type=float, default=None, help="Override the mode's default LR.")
    p.add_argument("--epochs", type=int, default=None, help="Override the mode's default epoch count.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override config batch size. Bigger = fewer iters/epoch = faster wall-clock "
                        "(this tiny model fits large batches). Consider raising --lr with it.")
    p.add_argument("--augment", type=int, default=1,
                   help="1 = rotation+noise augmentation on the train pipeline (held constant "
                        "across modes for a fair comparison); 0 = clean.")
    p.add_argument("--merge", type=str, default=None,
                   help="Comma-separated dataset:version specs to MERGE into ONE pretraining run, "
                        "e.g. 'camargo:10_20_dense_8cls,molinaro:10_20_both,scherpereel:10_20_both,"
                        "scherpereel_exo:10_20_both'. Each is split with the SAME seed and its "
                        "train+vali concatenated (test held out per dataset, so the downstream "
                        "per-dataset test split stays unseen). When set, the positional "
                        "dataset/version only pick the model config + save dir.")
    p.add_argument("--out_name", type=str, default=None,
                   help="Output ckpt name stem (default = --mode); final file is <stem>_seed<seed>.pt.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a seed whose output checkpoint already exists.")
    p.add_argument("--split", choices=["random", "group"], default="random",
                   help="random = legacy window shuffle; group = subject-grouped CV holdout.")
    p.add_argument("--group_label_index", type=int, default=1,
                   help="Label column holding the group/subject id (camargo: 1).")
    p.add_argument("--fold_id", type=int, default=0, help="Which CV fold (group split).")
    p.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (group split).")
    p.add_argument("--split_seed", type=int, default=3431,
                   help="Fixed seed defining the fold partition; independent of model seed.")
    p.add_argument("--holdout_dataset", type=str, default=None,
                   help="Under --split group: ONLY this dataset's test fold is held out of the "
                        "SSL pool (the one that will be evaluated). Other merged datasets are used "
                        "in full (never evaluated). Defaults to the positional dataset.")
    return p.parse_args()


def resolve_mode(args):
    d = MODE_DEFAULTS[args.mode]
    args.use_init = d["use_init"]
    args.lr = args.lr if args.lr is not None else d["lr"]
    args.epochs = args.epochs if args.epochs is not None else d["epochs"]
    args.out_name = args.out_name or args.mode
    if args.use_init and not args.model_file:
        sys.exit("--mode %s needs a starting checkpoint via -f/--model_file" % args.mode)
    if not args.use_init:
        args.model_file = None  # scratch: ignore any -f
    return args


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
    bs = args.batch_size if args.batch_size else base_train_cfg.batch_size
    train_cfg = base_train_cfg._replace(seed=seed, lr=args.lr, n_epochs=args.epochs, batch_size=bs)

    if args.split == "group":
        out_stem = os.path.join(save_dir, "%s_fold%d_seed%d" % (args.out_name, args.fold_id, seed))
    else:
        out_stem = os.path.join(save_dir, "%s_seed%d" % (args.out_name, seed))
    if args.skip_existing and os.path.exists(out_stem + ".pt"):
        print("Skipping existing checkpoint -> %s.pt" % out_stem)
        return out_stem + ".pt"

    set_seeds(seed)
    if args.merge:
        # Merge several datasets into ONE pretraining pool. Each is split with the
        # SAME seed (prepare_pretrain_dataset resets the RNG per call), so every
        # dataset's last-10% test split matches its bench_eval split and is held
        # out here -> no leakage. All datasets are (N, 20, 6), so they concatenate.
        specs = [s.strip() for s in args.merge.split(",") if s.strip()]
        holdout = args.holdout_dataset or args.dataset
        train_parts, vali_parts = [], []
        for spec in specs:
            ds, ver = spec.split(":")
            dpath = os.path.join(REPO_ROOT, "dataset", ds, "data_" + ver + ".npy")
            lpath = os.path.join(REPO_ROOT, "dataset", ds, "label_" + ver + ".npy")
            d = np.load(dpath).astype(np.float32)
            l = np.load(lpath).astype(np.float32)
            if args.split == "group" and ds != holdout:
                # Refinement B: this dataset is never evaluated -> use ALL its windows
                # for SSL, no fold holdout (only the holdout dataset's test fold is excluded).
                train_parts.append(d)
                print("  merge %-30s train=%6d vali=%6d (full, no holdout)" % (spec, d.shape[0], 0))
            else:
                dt, _, dv, _ = prepare_pretrain_dataset(
                    d, l, args.training_rate, seed=seed, split=args.split,
                    group_label_index=args.group_label_index, fold_id=args.fold_id,
                    n_folds=args.n_folds, split_seed=args.split_seed)
                train_parts.append(dt)
                vali_parts.append(dv)
                tag = "  [HOLDOUT fold %d/%d]" % (args.fold_id, args.n_folds) if args.split == "group" else ""
                print("  merge %-30s train=%6d vali=%6d%s" % (spec, dt.shape[0], dv.shape[0], tag))
        data_train = np.concatenate(train_parts, 0)
        data_vali = np.concatenate(vali_parts, 0) if vali_parts else data_train[:0]
    else:
        data = np.load(args.data_path).astype(np.float32)
        labels = np.load(args.label_path).astype(np.float32)

        # Same seed + same partition as the classifier path => identical train/vali/test
        # split. The test split is NOT returned here, so it is held out of pretraining.
        data_train, _, data_vali, _ = prepare_pretrain_dataset(
            data, labels, args.training_rate, seed=seed, split=args.split,
            group_label_index=args.group_label_index, fold_id=args.fold_id,
            n_folds=args.n_folds, split_seed=args.split_seed)

    norm = Preprocess4Normalization(args.model_cfg.feature_num)
    mask = Preprocess4Mask(mask_cfg)
    if args.augment:
        pipeline_train = [norm, Preprocess4Augment(args.model_cfg.feature_num), mask]
    else:
        pipeline_train = [norm, mask]
    pipeline_test = [norm, mask]  # recon loss is only a monitor -> keep it clean

    loader_train = DataLoader(LIBERTDataset4Pretrain(data_train, pipeline=pipeline_train),
                              shuffle=True, batch_size=train_cfg.batch_size)
    loader_vali = DataLoader(LIBERTDataset4Pretrain(data_vali, pipeline=pipeline_test),
                             shuffle=False, batch_size=train_cfg.batch_size)

    model = LIMUBertModel4Pretrain(args.model_cfg)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

    trainer = train.Trainer(train_cfg, model, optimizer, out_stem, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        return criterion(model(mask_seqs, masked_pos), seqs)

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        return model(mask_seqs, masked_pos), seqs

    def func_evaluate(seqs, predict_seqs):
        return criterion(predict_seqs, seqs).mean().cpu().numpy()

    print("\n=== %s seed=%d | train=%d vali=%d | init=%s | lr=%g epochs=%d augment=%d ==="
          % (args.mode, seed, data_train.shape[0], data_vali.shape[0],
             args.pretrain_model if args.use_init else "RANDOM",
             train_cfg.lr, train_cfg.n_epochs, int(bool(args.augment))))
    # model_file = the (stripped) starting ckpt for warmstart/dapt; None for scratch.
    trainer.pretrain(func_loss, func_forward, func_evaluate, loader_train, loader_vali,
                     model_file=args.pretrain_model)
    saved = out_stem + ".pt"
    print("Saved checkpoint -> %s" % saved)
    return saved


def main():
    args = parse_args()
    args = resolve_mode(args)
    args = build_io(args)
    base_train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    device = get_device(args.gpu)
    save_dir = os.path.dirname(args.save_path)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    produced = {}
    for seed in seeds:
        produced[seed] = pretrain_one_seed(args, base_train_cfg, mask_cfg, seed, save_dir, device)

    print("\n=== %s done. Per-seed checkpoints: ===" % args.mode)
    for seed, path in produced.items():
        print("  seed %-6d -> %s" % (seed, path))


if __name__ == "__main__":
    main()
