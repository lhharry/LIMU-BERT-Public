"""
Shared training-recipe helpers used by all classification paths (supervised /
bert-joint / bert-separated) so the three are configured identically.

The recipe is based on classifier.py's original training loop (sqrt-weighted
class-balanced CE, no rare-class filter) plus:
- early stopping
- an LR scale factor applied to `train_cfg.lr`
- optional linear warmup followed by cosine decay (build_scheduler)
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Recipe:
    early_stop_patience: int = 10
    lr_scale: float = 1
    warmup_epochs: int = 0          # linear warmup from 0 → base_lr; 0 disables
    cosine_decay: bool = False      # after warmup, cosine-decay to cosine_eta_min
    cosine_eta_min: float = 1e-6    # floor of cosine schedule (absolute LR, not multiplier)

    @staticmethod
    def default() -> "Recipe":
        return Recipe()


def build_scheduler(optimizer: torch.optim.Optimizer, recipe: "Recipe",
                    n_epochs: int) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Return a per-epoch LambdaLR implementing warmup+cosine, or None if neither is on.

    Multiplier convention: factor of base_lr. Cosine floor is (eta_min / base_lr)
    where base_lr comes from optimizer.param_groups[0]['lr'].
    """
    if recipe.warmup_epochs <= 0 and not recipe.cosine_decay:
        return None
    base_lr = optimizer.param_groups[0]["lr"]
    eta_ratio = recipe.cosine_eta_min / max(base_lr, 1e-12)
    w = max(recipe.warmup_epochs, 0)

    def lr_lambda(epoch: int) -> float:
        if epoch < w:
            return (epoch + 1) / max(w, 1)
        if not recipe.cosine_decay:
            return 1.0
        progress = (epoch - w) / max(n_epochs - w, 1)
        progress = min(max(progress, 0.0), 1.0)
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_criterion(label_train, label_num, device):
    """sqrt-weighted class-balanced cross-entropy (matches classifier.py)."""
    counts = np.bincount(label_train.flatten().astype(int), minlength=label_num)
    w = 1.0 / np.sqrt(counts.astype(float) + 1.0)
    w = w / w.sum() * label_num
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
