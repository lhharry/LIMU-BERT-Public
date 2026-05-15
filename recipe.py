"""
Shared training-recipe helpers used by both supervised and BERT benchmarking
paths so the two can be configured identically.

A Recipe bundles the choices that previously lived inside `bert_classify`
(class filter, class-weighted loss, cosine scheduler, early stopping, lr
scaling). Callers pass the same Recipe to `classify_benchmark` and
`bert_classify` to make the comparison apples-to-apples.

`Recipe.vanilla()` reproduces the original supervised behavior; `Recipe.filtered()`
reproduces the recipe that used to be hardcoded inside `bert_classify`.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Recipe:
    min_class_samples: int = 0          # 0 disables the rare-class filter
    class_weighted_loss: bool = False
    cosine_scheduler: bool = False
    cosine_eta_min: float = 1e-6
    early_stop_patience: Optional[int] = None
    lr_scale: float = 1.0               # multiplies train_cfg.lr

    @staticmethod
    def vanilla() -> "Recipe":
        return Recipe()

    @staticmethod
    def filtered() -> "Recipe":
        # Threshold lowered from 20 to 5 so the recipe still keeps classes at
        # very low label_rate (e.g. 0.01) where a per-class count of 20 is
        # impossible. filter_rare_classes also auto-falls-back to >=1 if the
        # threshold would erase every class.
        return Recipe(
            min_class_samples=5,
            class_weighted_loss=True,
            cosine_scheduler=True,
            early_stop_patience=10,
            lr_scale=0.1,
        )


def filter_rare_classes(splits, label_num, min_samples):
    """splits = (data_train, label_train, data_vali, label_vali, data_test, label_test).

    Drops classes whose train count < min_samples and remaps the remaining
    labels to a contiguous 0..K-1 range. Returns (new_splits, new_label_num,
    valid_classes). When min_samples <= 0 returns the inputs unchanged.
    """
    data_train, label_train, data_vali, label_vali, data_test, label_test = splits
    if min_samples <= 0:
        return splits, label_num, np.arange(label_num)

    counts = np.bincount(label_train.flatten().astype(int), minlength=label_num)
    valid = np.where(counts >= min_samples)[0]
    if len(valid) == 0:
        # Threshold wiped out every class (typical at very low label_rate).
        # Fall back to keeping any class that appears at all so we still get a
        # clean 0..K-1 label remap rather than an empty dataset.
        valid = np.where(counts >= 1)[0]
        print(f"[filter_rare_classes] threshold {min_samples} dropped all classes; "
              f"falling back to min_samples=1.")
    label_map = {old: new for new, old in enumerate(valid)}
    print("Keeping classes:", valid, "Dropping:", np.where(counts < min_samples)[0])

    def _f(d, l):
        mask = np.isin(l.flatten(), valid)
        new_l = np.array([label_map[x] for x in l.flatten()[mask]])
        return d[mask], new_l

    splits_f = (*_f(data_train, label_train),
                *_f(data_vali, label_vali),
                *_f(data_test, label_test))
    return splits_f, len(valid), valid


def make_criterion(label_train, label_num, device, class_weighted):
    if not class_weighted:
        return nn.CrossEntropyLoss()
    counts = np.bincount(label_train.flatten().astype(int), minlength=label_num)
    w = 1.0 / (counts.astype(float) + 1.0)         # Laplace smoothing for empty classes
    w = w / w.sum() * label_num
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))


def make_scheduler(optimizer, n_epochs, recipe: Recipe):
    if not recipe.cosine_scheduler:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=recipe.cosine_eta_min,
    )
