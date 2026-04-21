"""
compare.py — Compare δ² vs Adam training runs.

Reads the metrics.jsonl files from two experiments and produces
comparison plots and summary statistics.

Usage:
    python d2/experiments/compare.py \\
        --adam d2/output/adam_wikitext/metrics.jsonl \\
        --delta2 d2/output/delta2_wikitext/metrics.jsonl \\
        --output d2/experiments/comparison.png
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> list:
    """Load a metrics.jsonl file into a list of dicts."""
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def print_comparison(adam_metrics: list, d2_metrics: list):
    """Print a text comparison of the two runs."""

    def get_losses(metrics):
        return [m['loss'] for m in metrics if 'loss' in m]

    adam_losses = get_losses(adam_metrics)
    d2_losses = get_losses(d2_metrics)

    print("=" * 60)
    print("  Anamnesis-δ² vs Adam Comparison")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Adam':>12} {'δ²':>12}")
    print("-" * 54)

    if adam_losses and d2_losses:
        print(f"{'Final loss':<30} {adam_losses[-1]:>12.4f} {d2_losses[-1]:>12.4f}")
        print(f"{'Min loss':<30} {min(adam_losses):>12.4f} {min(d2_losses):>12.4f}")
        print(f"{'Steps':<30} {len(adam_losses):>12} {len(d2_losses):>12}")

    # Bassin stats (δ² only)
    d2_bassin = [m.get('bassin', {}) for m in d2_metrics if 'bassin' in m]
    if d2_bassin:
        last_bassin = d2_bassin[-1]
        print(f"\n{'δ² Bassin Stats (final step)':}")
        print(f"  Mean tension:     {last_bassin.get('abs_mean', 0):.6f}")
        print(f"  Max tension:      {last_bassin.get('max', 0):.4f}")
        print(f"  Nonzero fraction: {last_bassin.get('nonzero_frac', 0):.2%}")

    print()


def plot_comparison(adam_metrics: list, d2_metrics: list, output_path: str):
    """Generate comparison plots (if matplotlib is available)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        print("  pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss curves
    ax = axes[0]
    adam_steps = [m['step'] for m in adam_metrics if 'loss' in m]
    adam_losses = [m['loss'] for m in adam_metrics if 'loss' in m]
    d2_steps = [m['step'] for m in d2_metrics if 'loss' in m]
    d2_losses = [m['loss'] for m in d2_metrics if 'loss' in m]

    ax.plot(adam_steps, adam_losses, label='Adam (baseline)', alpha=0.7)
    ax.plot(d2_steps, d2_losses, label='δ² (experiment)', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss: Adam vs δ²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bassin tension (δ² only)
    ax = axes[1]
    d2_bassin_steps = [m['step'] for m in d2_metrics if 'bassin' in m]
    d2_tensions = [m['bassin'].get('abs_mean', 0) for m in d2_metrics if 'bassin' in m]

    if d2_tensions:
        ax.plot(d2_bassin_steps, d2_tensions, color='purple', label='Mean |tension|')
        ax.set_ylabel('Bassin Mean |Tension|')
        ax.set_title('δ² Tension Reservoir Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax.set_xlabel('Training Step')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare δ² vs Adam")
    parser.add_argument("--adam", required=True, help="Path to Adam metrics.jsonl")
    parser.add_argument("--delta2", required=True, help="Path to δ² metrics.jsonl")
    parser.add_argument("--output", default="d2/experiments/comparison.png",
                        help="Output plot path")
    args = parser.parse_args()

    adam_metrics = load_metrics(args.adam)
    d2_metrics = load_metrics(args.delta2)

    print_comparison(adam_metrics, d2_metrics)
    plot_comparison(adam_metrics, d2_metrics, args.output)
