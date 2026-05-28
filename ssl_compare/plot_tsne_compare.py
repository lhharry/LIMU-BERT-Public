#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Side-by-side t-SNE of BERT embeddings from the 4 in-domain SSL recipes:

  foundation : WANDS-HKUST limu_bert_x.pt           (cross-domain baseline)
  scratch    : in-domain SSL from random init       (this run)
  warmstart  : in-domain SSL warm-started on WANDS  (this run)
  dapt       : naive DAPT (gentle lr) on WANDS      (this run)

Same fixed sample of windows is fed through all 4 checkpoints so the panels
are visually comparable. Each panel is a TSNE(random_state=0) on the flattened
(seq_len * hidden) BERT outputs, colored by Camargo activity (label_index=0).

Run from the repo root (so relative paths to dataset/ and saved/ resolve):

  python ssl_compare/plot_tsne_compare.py
  python ssl_compare/plot_tsne_compare.py --seed 3431 --n_samples 1500 -g 0
  python ssl_compare/plot_tsne_compare.py --out ssl_compare/history/Run1_26052026_200epoch/tsne.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import load_dataset_label_names, load_dataset_stats, load_model_config
from models import LIMUBertModel4Pretrain
from utils import IMUDataset, Preprocess4Normalization, get_device, set_seeds

DATASET = "camargo"
DATASET_VERSION = "10_20_dense_8cls"
MODEL_VERSION = "v3"
LABEL_INDEX = 0   # 0 = activity for Camargo 8cls
PRETRAIN_DIR = os.path.join("saved", "pretrain_base_" + DATASET + "_" + DATASET_VERSION)

# (label, ckpt stem without .pt). Edit if you saved under different names.
CHECKPOINTS = [
    ("foundation (WANDS)", os.path.join(PRETRAIN_DIR, "limu_bert_x.pt")),
    ("scratch",            os.path.join(PRETRAIN_DIR, "scratch_seed3431.pt")),
    ("warmstart",          os.path.join(PRETRAIN_DIR, "warmstart_seed3431.pt")),
    ("dapt",               os.path.join(PRETRAIN_DIR, "dapt_seed3431.pt")),
]


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side t-SNE of SSL recipes.")
    p.add_argument("--seed", type=int, default=3431,
                   help="RNG seed for the sampled subset AND TSNE.random_state. "
                        "Same seed across runs -> identical panels.")
    p.add_argument("--n_samples", type=int, default=1500,
                   help="Windows to plot per panel. 1k-2k is usually enough.")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--perplexity", type=float, default=80.0,
                   help="TSNE perplexity. 30 is sklearn default; try 50 if blobs overlap.")
    p.add_argument("-g", "--gpu", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="If set, save to this path (PNG/PDF). Otherwise plt.show().")
    p.add_argument("--per_class_cap", type=int, default=None,
                   help="If set, cap per-class samples at this many (stratified sampling).")
    return p.parse_args()


def stratified_indices(labels_1d, n_samples, per_class_cap, rng):
    """Return a stable subset of indices, optionally balanced per class."""
    classes = np.unique(labels_1d)
    if per_class_cap is not None:
        chosen = []
        for c in classes:
            idx_c = np.where(labels_1d == c)[0]
            rng.shuffle(idx_c)
            chosen.append(idx_c[:per_class_cap])
        idx = np.concatenate(chosen)
        rng.shuffle(idx)
        return idx[:n_samples]
    idx = np.arange(labels_1d.shape[0])
    rng.shuffle(idx)
    return idx[:n_samples]


def load_data_and_model_cfg():
    model_cfg = load_model_config("pretrain_base", "base", MODEL_VERSION)
    if model_cfg is None:
        sys.exit("Cannot resolve model config base_%s" % MODEL_VERSION)
    dataset_cfg = load_dataset_stats(DATASET, DATASET_VERSION)
    if dataset_cfg is None:
        sys.exit("Cannot resolve dataset config %s_%s" % (DATASET, DATASET_VERSION))

    data_path = os.path.join("dataset", DATASET, "data_" + DATASET_VERSION + ".npy")
    label_path = os.path.join("dataset", DATASET, "label_" + DATASET_VERSION + ".npy")
    data = np.load(data_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)

    # Reshape to model.seq_len if the on-disk windows are longer (matches embedding.fetch_setup).
    if data.shape[1] != model_cfg.seq_len:
        merge = data.shape[1] // model_cfg.seq_len
        data = data.reshape(data.shape[0] * merge, model_cfg.seq_len, data.shape[2])
        labels = labels.reshape(labels.shape[0] * merge, model_cfg.seq_len, labels.shape[2])
    return data, labels, model_cfg, dataset_cfg


def embed_with_ckpt(ckpt_path, data_subset, labels_subset, model_cfg, device, batch_size):
    """Build a fresh LIMU-BERT, load ckpt, return (N, seq_len*hidden) numpy embeddings."""
    if not os.path.exists(ckpt_path):
        print("  !! missing ckpt, skipping: %s" % ckpt_path)
        return None
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    state = torch.load(ckpt_path, map_location="cpu")
    # Pretrain ckpts are full state dicts of LIMUBertModel4Pretrain (transformer +
    # fc/linear/norm/decoder). output_embed=True only uses transformer, so non-
    # transformer keys are unused but present -> strict=True is fine.
    model.load_state_dict(state)
    model.to(device).eval()

    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    loader = DataLoader(
        IMUDataset(data_subset, labels_subset, pipeline=pipeline),
        shuffle=False, batch_size=batch_size,
    )
    chunks = []
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)
            h = model(seqs)             # (B, seq_len, hidden)
            chunks.append(h.cpu().numpy())
    feat = np.concatenate(chunks, axis=0)
    feat = feat.reshape(feat.shape[0], -1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feat


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.gpu)

    data, labels, model_cfg, dataset_cfg = load_data_and_model_cfg()
    label_names, label_num = load_dataset_label_names(dataset_cfg, LABEL_INDEX)

    # Window-level label = first timestep's class (matches plot_embedding's convention).
    win_labels = labels[:, 0, LABEL_INDEX].astype(int)

    rng = np.random.RandomState(args.seed)
    idx = stratified_indices(win_labels, args.n_samples, args.per_class_cap, rng)
    data_sub = data[idx]
    labels_sub = labels[idx]
    win_labels_sub = win_labels[idx]
    print("Sampled %d windows; class distribution: %s"
          % (idx.shape[0], dict(zip(*np.unique(win_labels_sub, return_counts=True)))))

    # Compute t-SNE for each ckpt up-front, then plot at the end so a long TSNE run
    # doesn't block the figure window.
    panels = []
    for tag, ckpt in CHECKPOINTS:
        print("\n=== %s ===\n  ckpt: %s" % (tag, ckpt))
        feat = embed_with_ckpt(ckpt, data_sub, labels_sub, model_cfg, device, args.batch_size)
        if feat is None:
            panels.append((tag, None))
            continue
        print("  feat shape: %s -> running TSNE(perplexity=%s)" % (feat.shape, args.perplexity))
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
                    init="pca", learning_rate="auto")
        emb2d = tsne.fit_transform(feat)
        panels.append((tag, emb2d))

    classes = np.unique(win_labels_sub)
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    axes = axes[0]
    for ax, (tag, emb2d) in zip(axes, panels):
        ax.set_title(tag)
        ax.set_xticks([]); ax.set_yticks([])
        if emb2d is None:
            ax.text(0.5, 0.5, "ckpt missing", ha="center", va="center", transform=ax.transAxes)
            continue
        for ci, c in enumerate(classes):
            m = win_labels_sub == c
            name = label_names[int(c)] if label_names is not None and int(c) < len(label_names) else str(int(c))
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.7,
                       color=cmap(ci % cmap.N), label=name)

    # Shared legend on the right.
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="center right", fontsize=8,
                   bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.suptitle("t-SNE of LIMU-BERT embeddings (%s/%s, seed=%d, N=%d)"
                 % (DATASET, DATASET_VERSION, args.seed, idx.shape[0]),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=160, bbox_inches="tight")
        print("\nSaved figure -> %s" % args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
