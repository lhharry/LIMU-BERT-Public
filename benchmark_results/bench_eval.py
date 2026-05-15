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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["supervised", "bert"], required=True,
                   help="supervised = benchmark.py path (no pretrain), bert = classifier_bert.py path (with LIMU-BERT-X)")
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
    p.add_argument("--balance", type=int, default=1)
    p.add_argument("--frozen_bert", type=int, default=1)
    p.add_argument("--recipe", choices=["vanilla", "filtered"], default="filtered",
                   help="Training recipe applied identically to both supervised and bert paths. "
                        "'filtered' = drop classes with <20 train samples + class-weighted CE + cosine LR + early stop + lr*0.1. "
                        "'vanilla' = none of the above.")
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
    # (see config.py:load_model_config, which splits on '_'). If only one token
    # was supplied (e.g. "v1"), reuse it for both halves: "v1" -> "v1_v1".
    model_version = args_local.model_version
    if args_local.mode == "bert" and "_" not in model_version:
        model_version = model_version + "_" + model_version

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
        from utils import handle_argv
        from statistic import stat_results
        from recipe import Recipe

        recipe = Recipe.filtered() if args_local.recipe == "filtered" else Recipe.vanilla()

        if args_local.mode == "supervised":
            target = "bench_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            from benchmark import classify_benchmark
            label_test, preds = classify_benchmark(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                balance=bool(args_local.balance), method=args_local.method, recipe=recipe,
            )
        else:
            target = "bert_classifier_" + args_local.method
            args = handle_argv(target, tmp_basename, args_local.method)
            import classifier_bert
            classifier_bert.method = args_local.method  # see note: bert_classify uses free var
            label_test, preds = classifier_bert.bert_classify(
                args, args.label_index, args_local.training_rate, args_local.label_rate,
                frozen_bert=bool(args_local.frozen_bert), balance=bool(args_local.balance),
                recipe=recipe,
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
            "frozen_bert": bool(args_local.frozen_bert),
            "recipe": args_local.recipe,
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
