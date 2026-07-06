#!/usr/bin/env python
"""
Single-run worker for benchmarking. Invoked per (method, label_rate, seed) combo
by run_benchmark.py via subprocess. Trains one model, evaluates on the test set,
and writes acc/f1/confusion-matrix to --out_json.
"""
import argparse
import json
import os
import sys
import tempfile

# Make repo root importable when this file lives in benchmark_results/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import handle_argv
from statistic import stat_results
from recipe import Recipe
from benchmark import classify_benchmark
import classifier_bert
import embedding
import classifier as cls_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["supervised", "bert", "bert_separated"], required=True,
                   help="supervised = benchmark.py path (no pretrain); "
                        "bert = classifier_bert.py path (BERTClassifier joint, with frozen_bert flag); "
                        "bert_separated = embedding.py + classifier.py path (BERT eval/no_grad -> cached "
                        "embeddings -> standalone GRU head). All three modes use Recipe.default().")
    p.add_argument("--method", required=True,
                   help="gru / dcnn / deepsense / attn for supervised; base_gru / base_cnn / ... for bert")
    p.add_argument("--model_version", default="v1")
    p.add_argument("--dataset", default="camargo")
    p.add_argument("--dataset_version", default="10_20")
    p.add_argument("--label_rate", type=float, required=True)
    p.add_argument("--training_rate", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=3431)
    p.add_argument("--pretrain_model", default=None,
                   help="Path to pretrain .pt (only for --mode bert). Accepts relative or absolute.")
    p.add_argument("--save_model", default="bench_tmp")
    p.add_argument("--save_dir", default=None,
                   help="Folder name under saved/ to place this run's checkpoint "
                        "(e.g. 'bench_Run51_...'). When set, the trained model is saved to "
                        "saved/<save_dir>/<save_model>.pt instead of the per-target folder, so "
                        "concurrent benchmark runs don't overwrite each other.")
    p.add_argument("--label_index", type=int, default=-1)
    p.add_argument("--gpu", default=None)
    p.add_argument("--balance", type=int, default=0)
    p.add_argument("--frozen_bert", type=int, default=1)
    p.add_argument("--out_json", required=True)
    p.add_argument("--warmup_epochs", type=int, default=20,
                   help="Linear LR warmup over this many epochs (0 disables).")
    p.add_argument("--cosine_decay", type=int, default=1,
                   help="1 enables cosine LR decay after warmup; 0 disables.")
    p.add_argument("--cosine_eta_min", type=float, default=1e-6,
                   help="Absolute floor LR for cosine decay (used when --cosine_decay=1).")
    p.add_argument("--early_stop_patience", type=int, default=15,
                   help="Eval epochs without vali F1 improvement before stopping.")
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="Multiplier applied to train_cfg.lr (effective lr = train_cfg.lr * lr_scale).")
    p.add_argument("--split", choices=["random", "group"], default="random",
                   help="random = legacy window shuffle; group = subject-grouped CV fold.")
    p.add_argument("--group_label_index", type=int, default=1,
                   help="Label column holding the group/subject id (camargo: 1).")
    p.add_argument("--fold_id", type=int, default=0, help="Which CV fold (group split).")
    p.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (group split).")
    p.add_argument("--split_seed", type=int, default=3431,
                   help="Fixed seed defining the fold partition; independent of model seed.")
    p.add_argument("--n_epochs", type=int, default=None,
                   help="Override n_epochs from the base train config (e.g. fast smoke tests).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes for the downstream training loaders "
                        "(injected into the temp train config; 0 = main process).")
    args_local = p.parse_args()

    os.chdir(REPO_ROOT)

    # When --save_dir is given, all of this run's checkpoints land flat in
    # saved/<save_dir>/. os.makedirs (unlike config.py's single-level os.mkdir)
    # creates the folder; the branches below override args.save_path to point here.
    run_save_dir = None
    if args_local.save_dir:
        run_save_dir = os.path.join(REPO_ROOT, "saved", args_local.save_dir)
        os.makedirs(run_save_dir, exist_ok=True)

    base_cfg = "config/bert_classifier_train.json" if args_local.mode == "bert" else "config/train.json"
    with open(base_cfg, "r") as f:
        cfg = json.load(f)
    cfg["seed"] = args_local.seed
    cfg["num_workers"] = args_local.num_workers
    if args_local.n_epochs is not None:
        cfg["n_epochs"] = args_local.n_epochs
    tmp_dir = os.path.join(REPO_ROOT, "config")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=tmp_dir, prefix="bench_tmp_")
    json.dump(cfg, tmp)
    tmp.close()
    tmp_basename = os.path.basename(tmp.name)

    # BERT mode expects model_version as <bert_version>_<classifier_version>
    # (see config.py:load_model_config, which splits on '_'). Callers in
    # run_benchmark.py pass this explicitly (e.g. "v3_v1"); fail loudly if a
    # single token slips through so we don't silently fall back to base_v1.
    model_version = args_local.model_version
    if args_local.mode == "bert" and "_" not in model_version:
        raise SystemExit(
            f"--mode bert requires model_version='<bert_v>_<classifier_v>', got '{model_version}'"
        )

    sys.argv = [
        "bench_eval",
        model_version,
        args_local.dataset,
        args_local.dataset_version,
        "-t", "config/" + tmp_basename,
        "-s", args_local.save_model,
        "-l", str(args_local.label_index),
    ]
    if args_local.gpu is not None:
        sys.argv += ["-g", args_local.gpu]
    if args_local.pretrain_model is not None:
        sys.argv += ["-f", args_local.pretrain_model]

    try:
        recipe = Recipe(
            early_stop_patience=args_local.early_stop_patience,
            lr_scale=args_local.lr_scale,
            warmup_epochs=args_local.warmup_epochs,
            cosine_decay=bool(args_local.cosine_decay),
            cosine_eta_min=args_local.cosine_eta_min,
        )

        # Split config shared by all three paths (subject-grouped CV vs legacy random).
        split_kwargs = dict(
            split=args_local.split, group_label_index=args_local.group_label_index,
            fold_id=args_local.fold_id, n_folds=args_local.n_folds, split_seed=args_local.split_seed,
        )

        if args_local.mode == "supervised":
            target = "bench_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            if run_save_dir is not None:
                args.save_path = os.path.join(run_save_dir, args_local.save_model)
            label_test, preds = classify_benchmark(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                balance=bool(args_local.balance), method=args_local.method, recipe=recipe,
                **split_kwargs,
            )
        elif args_local.mode == "bert":
            target = "bert_classifier_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            if run_save_dir is not None:
                args.save_path = os.path.join(run_save_dir, args_local.save_model)
            classifier_bert.method = args_local.method  # see note: bert_classify uses free var
            label_test, preds = classifier_bert.bert_classify(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                frozen_bert=bool(args_local.frozen_bert), balance=bool(args_local.balance),
                recipe=recipe, **split_kwargs,
            )
        else:  # bert_separated
            if not args_local.pretrain_model:
                raise SystemExit("--pretrain_model is required for --mode bert_separated")

            # Stage 1: LIMU-BERT-X feature extractor (eval/no_grad inside Trainer.run).
            # Pretrain side uses base_v3 to match inference/test_csv.py.
            saved_argv = sys.argv[:]
            sys.argv = [
                "bench_eval", "v3",
                args_local.dataset, args_local.dataset_version,
                "-t", "config/pretrain.json",
                "-s", args_local.save_model,
                "-l", str(args_local.label_index),
                "-f", args_local.pretrain_model,
            ]
            if args_local.gpu is not None:
                sys.argv += ["-g", args_local.gpu]
            pre_args = handle_argv("pretrain_base", "pretrain.json", "base")

            _, embeddings, all_labels = embedding.generate_embedding_or_output(
                pre_args, save=False, output_embed=True
            )

            # Stage 2: standalone GRU head trained on cached embeddings via
            # classifier.classify_embeddings using the shared recipe. Method is
            # hardcoded to "gru"; the head version follows the requested
            # model_version so it stays aligned with the R-GRU / finetune paths.
            sys.argv = [
                "bench_eval", args_local.model_version,
                args_local.dataset, args_local.dataset_version,
                "-t", "config/" + tmp_basename,
                "-s", args_local.save_model,
                "-l", str(args_local.label_index),
            ]
            if args_local.gpu is not None:
                sys.argv += ["-g", args_local.gpu]
            cls_args = handle_argv("classifier_base_gru", tmp_basename, "gru")
            if run_save_dir is not None:
                cls_args.save_path = os.path.join(run_save_dir, args_local.save_model)
            sys.argv = saved_argv

            label_test, preds = cls_module.classify_embeddings(
                cls_args, embeddings, all_labels, cls_args.label_index,
                args_local.training_rate, args_local.label_rate,
                balance=bool(args_local.balance), method="gru", recipe=recipe,
                **split_kwargs,
            )

        acc, matrix, f1 = stat_results(label_test, preds)
        result = {
            "acc": float(acc),
            "f1": float(f1),
            "mode": args_local.mode,
            "method": args_local.method,
            "model_version": args_local.model_version,
            "dataset": args_local.dataset,
            "dataset_version": args_local.dataset_version,
            "label_rate": args_local.label_rate,
            "training_rate": args_local.training_rate,
            "seed": args_local.seed,
            "pretrain_model": args_local.pretrain_model,
            "frozen_bert": (None if args_local.mode == "bert_separated"
                            else bool(args_local.frozen_bert)),
            "recipe": {
                "early_stop_patience": recipe.early_stop_patience,
                "lr_scale": recipe.lr_scale,
                "warmup_epochs": recipe.warmup_epochs,
                "cosine_decay": recipe.cosine_decay,
                "cosine_eta_min": recipe.cosine_eta_min,
            },
            "split": {
                "mode": args_local.split,
                "group_label_index": args_local.group_label_index,
                "fold_id": args_local.fold_id,
                "n_folds": args_local.n_folds,
                "split_seed": args_local.split_seed,
            },
            "confusion_matrix": matrix.tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args_local.out_json)), exist_ok=True)
        with open(args_local.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[BENCH_RESULT] acc={acc:.4f} f1={f1:.4f}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    main()
