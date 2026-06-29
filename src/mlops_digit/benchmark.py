"""Inference latency benchmark for the trained models.

Measures in-process prediction latency (model compute only, no network) for
single-sample requests and a batch, reporting mean / p50 / p90 / p95 / p99
latency and throughput. Results are saved to JSON and a bar chart.

Usage:
    python -m mlops_digit.benchmark
    python -m mlops_digit.benchmark --runs balanced --iterations 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlops_digit.data import make_splits  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _percentiles(latencies_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(latencies_ms.max()),
    }


def benchmark_model(
    model_path: Path,
    X_single: np.ndarray,
    X_batch: np.ndarray,
    iterations: int,
    warmup: int,
) -> dict:
    model = joblib.load(model_path)

    # Warm up (first calls pay one-off costs).
    for _ in range(warmup):
        model.predict(X_single)

    single_lat = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        start = time.perf_counter()
        model.predict(X_single)
        single_lat[i] = (time.perf_counter() - start) * 1000.0

    batch_iters = max(iterations // 10, 10)
    batch_lat = np.empty(batch_iters, dtype=np.float64)
    for i in range(batch_iters):
        start = time.perf_counter()
        model.predict(X_batch)
        batch_lat[i] = (time.perf_counter() - start) * 1000.0

    single_stats = _percentiles(single_lat)
    batch_stats = _percentiles(batch_lat)
    return {
        "model": model_path.stem,
        "single": {
            **single_stats,
            "throughput_rps": 1000.0 / single_stats["mean_ms"],
        },
        "batch": {
            **batch_stats,
            "batch_size": int(X_batch.shape[0]),
            "throughput_rps": (X_batch.shape[0] * 1000.0) / batch_stats["mean_ms"],
        },
    }


def plot_latency(results: list[dict], out_path: Path) -> None:
    names = [r["model"] for r in results]
    p50 = [r["single"]["p50_ms"] for r in results]
    p95 = [r["single"]["p95_ms"] for r in results]
    p99 = [r["single"]["p99_ms"] for r in results]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, p50, width, label="p50", color="#4C72B0")
    ax.bar(x, p95, width, label="p95", color="#DD8452")
    ax.bar(x + width, p99, width, label="p99", color="#C44E52")
    ax.set_ylabel("Single-sample latency (ms)")
    ax.set_title("Inference latency by model (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model inference latency.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Model names to benchmark (default: all .joblib files in models/).",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    splits = make_splits()
    X_single = splits.X_test[:1]
    X_batch = splits.X_test[: args.batch_size]

    if args.runs:
        model_paths = [MODELS_DIR / f"{name}.joblib" for name in args.runs]
    else:
        model_paths = sorted(MODELS_DIR.glob("*.joblib"))
    model_paths = [p for p in model_paths if p.exists()]
    if not model_paths:
        raise SystemExit(
            "No models found. Run training first (python -m mlops_digit.analyze)."
        )

    results = [
        benchmark_model(p, X_single, X_batch, args.iterations, args.warmup)
        for p in model_paths
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_latency(results, FIGURES_DIR / "latency_benchmark.png")
    with open(ARTIFACTS_DIR / "latency_benchmark.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "iterations": args.iterations}, fh, indent=2)

    header = (
        f"{'model':<12}{'p50_ms':>9}{'p95_ms':>9}{'p99_ms':>9}"
        f"{'rps':>10}{'batch_rps':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<12}"
            f"{r['single']['p50_ms']:>9.3f}{r['single']['p95_ms']:>9.3f}"
            f"{r['single']['p99_ms']:>9.3f}{r['single']['throughput_rps']:>10.0f}"
            f"{r['batch']['throughput_rps']:>12.0f}"
        )
    print(f"\nResults written to: {ARTIFACTS_DIR / 'latency_benchmark.json'}")


if __name__ == "__main__":
    main()
