#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : utils.py
# @Description :

import argparse

from scipy.interpolate import CubicSpline
from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config


""" Utils Functions """

import random

import numpy as np
import torch
import sys


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all', extra=None):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    if extra is not None:
        # Filter a secondary per-window array (e.g. subject id) by the SAME keep
        # mask derived from `label`, so it stays aligned to the returned windows.
        # extra is reshaped like `label` (n_windows, merge); subject is constant
        # within a kept window, so take column 0.
        return data[index], np.array(label_new), extra[index, 0]
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


def prepare_pretrain_dataset(data, labels, training_rate, seed=None,
                             split="random", group_label_index=1,
                             fold_id=0, n_folds=5, split_seed=3431):
    set_seeds(seed)
    if split == "group":
        # grouped CV: held-out test groups are excluded so SSL never sees them.
        data_train, label_train, data_vali, label_vali, data_test, label_test \
            = partition_grouped_and_reshape(data, labels, label_index=0,
                                            group_label_index=group_label_index, change_shape=False,
                                            fold_id=fold_id, n_folds=n_folds, split_seed=split_seed)
    else:
        data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                      , training_rate=training_rate, vali_rate=0.1
                                                                                                      , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False
                               , split="random", group_label_index=1, fold_id=0, n_folds=5, split_seed=3431):

    set_seeds(seed)
    # When balancing on a multi-subject dataset, also balance across subjects
    # (equal windows per subject x class). Falls back to class-only balance for
    # single-subject configs or when the group column coincides with the label.
    subject_balance = (
        balance and group_label_index is not None
        and group_label_index != label_index
        and group_label_index < labels.shape[2]
        and np.unique(labels[:, 0, group_label_index]).size > 1
    )
    if split == "group":
        parts = partition_grouped_and_reshape(data, labels, label_index=label_index,
                                              group_label_index=group_label_index, change_shape=change_shape,
                                              merge=merge, merge_mode=merge_mode,
                                              fold_id=fold_id, n_folds=n_folds, split_seed=split_seed,
                                              return_group=subject_balance)
    else:
        parts = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1
                                      , change_shape=change_shape, merge=merge, merge_mode=merge_mode
                                      , group_label_index=group_label_index, return_group=subject_balance)
    if subject_balance:
        data_train, label_train, data_vali, label_vali, data_test, label_test, group_train = parts
    else:
        data_train, label_train, data_vali, label_vali, data_test, label_test = parts
    set_seeds(seed)
    if balance:
        if subject_balance:
            data_train_label, label_train_label, _, _ \
                = prepare_simple_dataset_balance_grouped(data_train, label_train, group_train, training_rate=label_rate)
        else:
            data_train_label, label_train_label, _, _ \
                = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True
                          , group_label_index=None, return_group=False):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    # Optionally carry the group/subject id for the TRAIN windows through the exact
    # same reshape+merge so it stays aligned to data_train (used for subject-balanced
    # sampling). group id is constant within a window.
    group_train = labels[:train_num, ..., group_label_index] if return_group else None
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
        if return_group:
            group_train = reshape_label(group_train, merge)
    if change_shape and merge != 0:
        if return_group:
            data_train, label_train, group_train = merge_dataset(
                data_train, label_train, mode=merge_mode, extra=group_train)
        else:
            data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    if return_group:
        return data_train, label_train, data_vali, label_vali, data_test, label_test, group_train
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def grouped_fold_assignment(groups, fold_id, n_folds, split_seed):
    """Deterministically assign unique group ids to CV folds.

    The fold definition depends ONLY on (sorted unique groups, split_seed, n_folds)
    -- NOT on the per-run model seed -- so the same fold is reproduced by pretrain
    and downstream, and across model seeds/label_rates. Returns (train_groups,
    vali_groups, test_groups) as python sets. test = fold[fold_id],
    vali = fold[(fold_id + 1) % n_folds], train = the remaining folds.
    """
    if not (0 <= fold_id < n_folds):
        sys.exit("fold_id %d out of range [0, %d)" % (fold_id, n_folds))
    uniq = np.unique(groups)
    if len(uniq) < n_folds:
        sys.exit("n_folds=%d but only %d unique groups" % (n_folds, len(uniq)))
    # local RNG -> does not touch the global np.random state used by the
    # balanced label sampler (which is seeded by the model seed elsewhere).
    rng = np.random.RandomState(split_seed)
    uniq = uniq.copy()
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    test_groups = set(folds[fold_id].tolist())
    vali_groups = set(folds[(fold_id + 1) % n_folds].tolist())
    train_groups = set(uniq.tolist()) - test_groups - vali_groups
    return train_groups, vali_groups, test_groups


def partition_grouped_and_reshape(data, labels, label_index=0, group_label_index=1,
                                  change_shape=True, merge=0, merge_mode='all',
                                  fold_id=0, n_folds=5, split_seed=3431,
                                  return_group=False):
    """Subject-grouped k-fold partition: no group (e.g. subject) appears in more
    than one of train/vali/test. Drop-in shape-compatible with
    partition_and_reshape (same 6-tuple return, or 7-tuple with the aligned
    train group ids when return_group=True), but the split is grouped CV
    instead of a random window shuffle. Grouping is read from labels[:, 0,
    group_label_index] (group id is constant within a window)."""
    groups = labels[:, 0, group_label_index].astype(int)
    train_groups, vali_groups, test_groups = grouped_fold_assignment(
        groups, fold_id, n_folds, split_seed)

    train_mask = np.isin(groups, list(train_groups))
    vali_mask = np.isin(groups, list(vali_groups))
    test_mask = np.isin(groups, list(test_groups))

    data_train, data_vali, data_test = data[train_mask], data[vali_mask], data[test_mask]
    t = np.min(labels[:, :, label_index])
    label_train = labels[train_mask, ..., label_index] - t
    label_vali = labels[vali_mask, ..., label_index] - t
    label_test = labels[test_mask, ..., label_index] - t
    # Carry the subject id for TRAIN windows through the same reshape+merge (see
    # partition_and_reshape); needed for subject-balanced sampling under grouped CV.
    group_train = labels[train_mask, ..., group_label_index] if return_group else None
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
        if return_group:
            group_train = reshape_label(group_train, merge)
    if change_shape and merge != 0:
        if return_group:
            data_train, label_train, group_train = merge_dataset(
                data_train, label_train, mode=merge_mode, extra=group_train)
        else:
            data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Grouped CV fold %d/%d (split_seed=%d) | train groups=%s vali=%s test=%s'
          % (fold_id, n_folds, split_seed, sorted(train_groups), sorted(vali_groups), sorted(test_groups)))
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    if return_group:
        return data_train, label_train, data_vali, label_vali, data_test, label_test, group_train
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    # labels arrive already zero-based from partition_*; a nonzero min means class 0
    # is absent from this train pool and subtracting would silently shift every label.
    t = np.min(labels)
    assert t == 0, "prepare_simple_dataset: train pool min label is %s (class 0 missing); refusing to re-shift labels" % t
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    # same contract as prepare_simple_dataset: input labels must already be zero-based
    t = np.min(labels)
    assert t == 0, "prepare_simple_dataset_balance: train pool min label is %s (class 0 missing); refusing to re-shift labels" % t
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance_grouped(data, labels, groups, training_rate=0.8):
    """Like prepare_simple_dataset_balance, but balances across SUBJECTS as well as
    classes: draws an equal number of windows from every (subject x class) cell, so
    each subject contributes equally to each activity in the labeled pool. `groups`
    is the per-window subject id aligned to `data`/`labels` (see partition_*'s
    return_group). Empty (subject, class) cells (a subject that never did an activity)
    are skipped and do not drag the per-cell count to zero. Same zero-based-label
    contract as prepare_simple_dataset_balance."""
    labels = labels.astype(int)
    groups = groups.astype(int)
    classes = np.unique(labels)
    subjects = np.unique(groups)
    n_classes, n_subjects = classes.size, subjects.size

    cells = {}                       # (subject, class) -> flat window indices
    nonempty_counts = []
    for s in subjects:
        for c in classes:
            idx = np.argwhere((groups == s) & (labels == c)).reshape(-1)
            cells[(s, c)] = idx
            if idx.size > 0:
                nonempty_counts.append(idx.size)
    n_empty = n_subjects * n_classes - len(nonempty_counts)
    smallest_cell = min(nonempty_counts) if nonempty_counts else 0
    per_cell = min(smallest_cell, int(data.shape[0] * training_rate / (n_classes * n_subjects)))
    if per_cell == smallest_cell and smallest_cell > 0:
        print("Warning! You are using all windows of the smallest (subject,class) cell.")

    index = np.zeros(data.shape[0], dtype=bool)
    for idx in cells.values():
        if idx.size == 0:
            continue
        np.random.shuffle(idx)
        index[idx[:per_cell]] = True

    # same contract as prepare_simple_dataset_balance: input labels already zero-based
    t = np.min(labels)
    assert t == 0, "prepare_simple_dataset_balance_grouped: train pool min label is %s (class 0 missing); refusing to re-shift labels" % t
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance(subject) Label Size: %d, Unlabel Size: %d; subjects=%d classes=%d '
          'per(subject,class)=%d empty_cells=%d; Real Label Rate: %0.3f'
          % (label_train.shape[0], label_test.shape[0], n_subjects, n_classes, per_cell,
             n_empty, label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Augment(Pipeline):
    """ Accel-SimCLR style augmentation pack for masked-reconstruction pretraining.
        Placed before Preprocess4Mask -> both the masked input and the reconstruction
        target are augmented (stochastic view, not a denoising objective).

        The 6 transforms follow Tang et al. 2020 (the set RelCon's distance network
        uses for accel-semantic invariance). All probabilities are independent, so
        most calls apply 1-2 transforms rather than all of them.

        Args (all defaults are gentle enough that rotate+noise still dominates,
        matching prior runs; tune p_* to enable more aggressive mixes):
          rotate           : random uniform 3D rotation per 3-axis sensor group
          noise            : per-sample Gaussian jitter (applied last)
          scale            : per-axis multiplicative scalar ~ N(1, scale_std)
          mag_warp         : smooth per-channel magnitude curve (cubic spline knots)
          time_warp        : smooth temporal resampling (monotone cubic spline)
          permute          : split window into K segments, randomly reorder
          channel_shuffle  : permute the 3 axes inside each sensor group
                             (semantics-destroying; default OFF) """
    def __init__(self, feature_len,
                 rotate=True, p_rotate=0.5,
                 noise=True, noise_std=0.02,
                 scale=True, p_scale=0.5, scale_std=0.1,
                 mag_warp=True, p_mag_warp=0.3, mag_warp_std=0.2, mag_warp_knots=4,
                 time_warp=True, p_time_warp=0.3, time_warp_std=0.2, time_warp_knots=4,
                 permute=True, p_permute=0.2, permute_segments=4,
                 channel_shuffle=False, p_channel_shuffle=0.0):
        super().__init__()
        self.feature_len = feature_len
        self.rotate = rotate
        self.p_rotate = p_rotate
        self.noise = noise
        self.noise_std = noise_std
        self.scale = scale
        self.p_scale = p_scale
        self.scale_std = scale_std
        self.mag_warp = mag_warp
        self.p_mag_warp = p_mag_warp
        self.mag_warp_std = mag_warp_std
        self.mag_warp_knots = mag_warp_knots
        self.time_warp = time_warp
        self.p_time_warp = p_time_warp
        self.time_warp_std = time_warp_std
        self.time_warp_knots = time_warp_knots
        self.permute = permute
        self.p_permute = p_permute
        self.permute_segments = permute_segments
        self.channel_shuffle = channel_shuffle
        self.p_channel_shuffle = p_channel_shuffle

    @staticmethod
    def _random_rotation():
        # Uniform random 3D rotation (Rodrigues): random axis + random angle
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(0, 2 * np.pi)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def _apply_rotation(self, inst):
        R = self._random_rotation()
        for s in range(0, self.feature_len - self.feature_len % 3, 3):
            inst[:, s:s + 3] = inst[:, s:s + 3] @ R.T
        return inst

    def _apply_scaling(self, inst):
        # Per-axis multiplicative scalar; one draw per channel (matches Tang et al.).
        factors = np.random.normal(1.0, self.scale_std, self.feature_len)
        inst[:, :self.feature_len] *= factors[None, :]
        return inst

    def _smooth_curve(self, T, n_knots, std, anchor=1.0):
        # Cubic spline through n_knots equally-spaced control points; values ~ N(anchor, std).
        knot_x = np.linspace(0, T - 1, n_knots + 2)
        knot_y = np.random.normal(anchor, std, n_knots + 2)
        cs = CubicSpline(knot_x, knot_y)
        return cs(np.arange(T))

    def _apply_mag_warp(self, inst):
        T = inst.shape[0]
        for c in range(self.feature_len):
            curve = self._smooth_curve(T, self.mag_warp_knots, self.mag_warp_std, anchor=1.0)
            inst[:, c] *= curve
        return inst

    def _apply_time_warp(self, inst):
        # Build a monotone time-warp via cumulative normalization of a positive smooth curve.
        T = inst.shape[0]
        speed = self._smooth_curve(T, self.time_warp_knots, self.time_warp_std, anchor=1.0)
        speed = np.clip(speed, 1e-3, None)
        cum = np.cumsum(speed)
        warped_idx = (cum - cum[0]) / (cum[-1] - cum[0]) * (T - 1)
        orig_idx = np.arange(T)
        for c in range(self.feature_len):
            inst[:, c] = np.interp(orig_idx, warped_idx, inst[:, c])
        return inst

    def _apply_permute(self, inst):
        T = inst.shape[0]
        k = min(self.permute_segments, T)
        if k <= 1:
            return inst
        # Random split points; segment lengths >= 1.
        cuts = sorted(np.random.choice(np.arange(1, T), size=k - 1, replace=False))
        segs = np.split(inst, cuts, axis=0)
        order = np.random.permutation(len(segs))
        inst_new = np.concatenate([segs[i] for i in order], axis=0)
        # Only mutate the feature columns; any extra cols (labels etc.) are untouched
        # because instance is feature-only here (see IMUDataset / LIBERTDataset4Pretrain).
        return inst_new

    def _apply_channel_shuffle(self, inst):
        for s in range(0, self.feature_len - self.feature_len % 3, 3):
            perm = np.random.permutation(3)
            inst[:, s:s + 3] = inst[:, s:s + 3][:, perm]
        return inst

    def __call__(self, instance):
        inst = instance.copy()
        if self.rotate and np.random.rand() < self.p_rotate:
            inst = self._apply_rotation(inst)
        if self.scale and np.random.rand() < self.p_scale:
            inst = self._apply_scaling(inst)
        if self.mag_warp and np.random.rand() < self.p_mag_warp:
            inst = self._apply_mag_warp(inst)
        if self.time_warp and np.random.rand() < self.p_time_warp:
            inst = self._apply_time_warp(inst)
        if self.permute and np.random.rand() < self.p_permute:
            inst = self._apply_permute(inst)
        if self.channel_shuffle and np.random.rand() < self.p_channel_shuffle:
            inst = self._apply_channel_shuffle(inst)
        # Noise applied last so it isn't smoothed away by warp/permute.
        if self.noise:
            fl = self.feature_len
            inst[:, :fl] += np.random.normal(0, self.noise_std, inst[:, :fl].shape)
        return inst


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)


class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)


class LIBERTDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return len(self.data)


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_version', type=str, help='Model config')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'camargo', 'molinaro', 'scherpereel', 'scherpereel_exo','jetson_leg','merged'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120', '10_20', '10_60', '10_20_dense', '10_20_dense_8cls', '10_20_both','10_20_dense_9cls',
                                                                                       '10_20_both_dense_9cls','10_20_both_xyz_leg','10_20_both_xyz_pocket','10_20_9cls_align',
                                                                                       '10_20_both_01_xyz_pocket', '10_20_both_01_xyz_leg','10_20_both_02_xyz_pocket', '10_20_both_02_xyz_leg',
                                                                                       '10_20_both_03_xyz_leg','10_20_both_03_xyz_pocket','10_20_both_0102_xyz_leg','10_20_both_0102_xyz_pocket'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model',
                        help='The saved model name')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
    return args



def handle_argv_simple():
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'camargo', 'merge'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120'])
    args = parser.parse_args()
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    return args


def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
