"""
Shared training-recipe helpers used by all classification paths (supervised /
bert-joint / bert-separated) so the three are configured identically.

The recipe is based on classifier.py's original training loop (sqrt-weighted
class-balanced CE, no scheduler, no rare-class filter) plus two additions:
early stopping and an LR scale factor applied to `train_cfg.lr`.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Recipe:
    early_stop_patience: int = 10
    lr_scale: float = 0.1

    @staticmethod
    def default() -> "Recipe":
        return Recipe()


def make_criterion(label_train, label_num, device):
    """sqrt-weighted class-balanced cross-entropy (matches classifier.py)."""
    counts = np.bincount(label_train.flatten().astype(int), minlength=label_num)
    w = 1.0 / np.sqrt(counts.astype(float) + 1.0)
    w = w / w.sum() * label_num
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
