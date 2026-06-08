#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train
from config import MaskConfig, TrainConfig, PretrainModelConfig, load_model_config, load_dataset_stats, create_io_config
from models import LIMUBertModel4Pretrain
from utils import (set_seeds, get_device, LIBERTDataset4Pretrain,
                   load_pretrain_data_config, prepare_pretrain_dataset,
                   Preprocess4Normalization, Preprocess4Mask)


MODE = "base"  # mirrors the hardcoded mode in pretrain.py


def handle_argv_finetune():
    parser = argparse.ArgumentParser(description='Fine-tune a pretrained LIMU-BERT model (masked reconstruction only)')
    parser.add_argument('model_version', type=str, help='Model config version (e.g. v1, v2, v3)')
    parser.add_argument('dataset', type=str, help='Dataset name',
                        choices=['hhar', 'motion', 'uci', 'shoaib', 'camargo'])
    parser.add_argument('dataset_version', type=str, help='Dataset version',
                        choices=['10_100', '20_120', '10_20', '10_60'])
    parser.add_argument('pretrain_model', type=str,
                        help='Path to the pretrained model file (with or without .pt extension)')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='GPU index to use')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/pretrain.json',
                        help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask config json file path')
    parser.add_argument('-s', '--save_model', type=str, default='model_finetune',
                        help='Output model filename (saved under saved/ directory)')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze the embedding layer during fine-tuning')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Freeze transformer attention/FF layers (any value >0 freezes the shared weights)')
    parser.add_argument('--lr_scale', type=float, default=0.1,
                        help='Multiply the LR from train_cfg by this factor (default 0.1 = 10x smaller than pretrain LR)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    target = 'pretrain_' + MODE
    # load_model_config builds the key as MODE + "_" + model_version, e.g. "base_v1"
    model_cfg = load_model_config(target, MODE, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg

    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg

    args = create_io_config(args, args.dataset, args.dataset_version,
                            pretrain_model=args.pretrain_model, target=target)
    return args


def freeze_pretrain_layers(model, freeze_embed=False, freeze_layers=0):
    transformer = model.transformer
    if freeze_embed:
        for p in transformer.embed.parameters():
            p.requires_grad = False
        print("Embedding layer frozen.")

    if freeze_layers > 0:
        # The Transformer uses parameter sharing so there is only one set of
        # attn/proj/norm/pwff weights regardless of n_layers. Freezing them
        # means those shared weights won't be updated.
        for p in transformer.attn.parameters():
            p.requires_grad = False
        for p in transformer.proj.parameters():
            p.requires_grad = False
        for p in transformer.norm1.parameters():
            p.requires_grad = False
        for p in transformer.pwff.parameters():
            p.requires_grad = False
        for p in transformer.norm2.parameters():
            p.requires_grad = False
        print("Transformer attention/FF layers frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Trainable parameters: %d / %d" % (trainable, total))


def main(args, training_rate=0.8):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    print("mask_cfg(%s): ratio=%g alpha=%g max_gram=%g prob=%g replace=%g"
          % (args.mask_cfg, mask_cfg.mask_ratio, mask_cfg.mask_alpha,
             mask_cfg.max_gram, mask_cfg.mask_prob, mask_cfg.replace_prob))

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate,
                                                                               seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    model = LIMUBertModel4Pretrain(model_cfg)

    # Load pretrained weights — required for fine-tuning
    print("Loading pretrained model from:", args.pretrain_model)
    model.load_state_dict(torch.load(args.pretrain_model + '.pt', map_location='cpu'))

    freeze_pretrain_layers(model,
                           freeze_embed=args.freeze_embed,
                           freeze_layers=args.freeze_layers)

    finetune_lr = train_cfg.lr * args.lr_scale
    print("Pretrain LR: %.2e  →  Fine-tune LR: %.2e  (lr_scale=%.3f)" % (train_cfg.lr, finetune_lr, args.lr_scale))

    criterion = nn.MSELoss(reduction='none')
    device = get_device(args.gpu)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return criterion(seq_recon, seqs)

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        return criterion(predict_seqs, seqs).mean().cpu().numpy()

    # Pass model_file=None: weights are already loaded above, trainer.pretrain
    # would overwrite them if we passed a path here.
    trainer.pretrain(func_loss, func_forward, func_evaluate,
                     data_loader_train, data_loader_test, model_file=None)


if __name__ == "__main__":
    args = handle_argv_finetune()
    training_rate = 0.8
    main(args, training_rate)
