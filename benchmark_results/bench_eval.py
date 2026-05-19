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
    p.add_argument("--label_index", type=int, default=-1)
    p.add_argument("--gpu", default=None)
    p.add_argument("--balance", type=int, default=0)
    p.add_argument("--frozen_bert", type=int, default=1)
    p.add_argument("--out_json", required=True)
    args_local = p.parse_args()

    os.chdir(REPO_ROOT)

    base_cfg = "config/bert_classifier_train.json" if args_local.mode == "bert" else "config/train.json"
    with open(base_cfg, "r") as f:
        cfg = json.load(f)
    cfg["seed"] = args_local.seed
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
        recipe = Recipe.default()

        if args_local.mode == "supervised":
            target = "bench_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            label_test, preds = classify_benchmark(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                balance=bool(args_local.balance), method=args_local.method, recipe=recipe,
            )
        elif args_local.mode == "bert":
            target = "bert_classifier_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            classifier_bert.method = args_local.method  # see note: bert_classify uses free var
            label_test, preds = classifier_bert.bert_classify(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                frozen_bert=bool(args_local.frozen_bert), balance=bool(args_local.balance),
                recipe=recipe,
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
            # classifier.classify_embeddings using the shared Recipe.default().
            # Method is hardcoded to "gru" because separated mode = gru_v1 head.
            sys.argv = [
                "bench_eval", "v1",
                args_local.dataset, args_local.dataset_version,
                "-t", "config/" + tmp_basename,
                "-s", args_local.save_model,
                "-l", str(args_local.label_index),
            ]
            if args_local.gpu is not None:
                sys.argv += ["-g", args_local.gpu]
            cls_args = handle_argv("classifier_base_gru", tmp_basename, "gru")
            sys.argv = saved_argv

            label_test, preds = cls_module.classify_embeddings(
                cls_args, embeddings, all_labels, cls_args.label_index,
                args_local.training_rate, args_local.label_rate,
                balance=bool(args_local.balance), method="gru", recipe=recipe,
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
            "recipe": "default",
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
