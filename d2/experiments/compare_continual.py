"""
compare_continual.py — Side-by-side comparison of continual-learning runs.

WHAT THIS DOES, FOR THE DUMMIES:
=================================

After running `continual.py` for several methods (adam, ewc, gem, delta2,
controller), you get one JSON result file per run. This script ingests
all of them and prints a comparison table:

      Method     ACC      BWT       FWT     Duration
      ─────────────────────────────────────────────────
      adam       0.4521  -0.4128   +0.0231   45s
      ewc        0.7234  -0.0892   +0.0312   58s
      gem        0.7681  -0.0534   +0.0287   91s
      delta2     0.7102  -0.1124   +0.0398   52s
      controller 0.7891  -0.0421   +0.0445   76s   ← winner if BWT close to 0

The headline number is BWT (backward transfer). BWT close to zero means
"no forgetting." Negative means "forgot earlier tasks." This is what
δ² is supposed to win on.

ACC matters too (overall accuracy across all tasks at the end of training)
but a method can have decent ACC while still forgetting a lot — it's
backward transfer that tells the real story.


USAGE:
=======

    python d2/experiments/compare_continual.py \\
        d2/output/continual_adam.json \\
        d2/output/continual_ewc.json \\
        d2/output/continual_gem.json \\
        d2/output/continual_delta2.json \\
        d2/output/continual_controller.json
"""

import argparse
import json
from pathlib import Path


def load_result(path: str) -> dict:
    """Load one continual.py result file."""
    with open(path) as f:
        return json.load(f)


def print_table(results: list[dict]):
    """Print a side-by-side comparison."""
    print()
    print("=" * 78)
    print(f"  Continual-learning benchmark: {results[0].get('benchmark', '?')}, "
          f"{results[0].get('tasks', '?')} tasks, "
          f"{results[0].get('epochs_per_task', '?')} epochs/task")
    print("=" * 78)

    # Header
    print(f"  {'METHOD':<14} {'ACC':>8} {'BWT':>10} {'FWT':>10} {'TIME(s)':>10}")
    print("  " + "─" * 56)

    # Sort: best BWT first (BWT close to 0 = least forgetting)
    sorted_results = sorted(results, key=lambda r: -r.get('bwt', -1))

    for r in sorted_results:
        method = r.get('method', '?')
        acc = r.get('acc', 0.0)
        bwt = r.get('bwt', 0.0)
        fwt = r.get('fwt', 0.0)
        dur = r.get('duration_seconds', 0.0)
        print(f"  {method:<14} {acc:>8.4f} {bwt:>+10.4f} {fwt:>+10.4f} {dur:>10.1f}")

    print("=" * 78)
    print()
    print("  ACC: average accuracy across all tasks at end of training (higher better)")
    print("  BWT: backward transfer — average accuracy drop on earlier tasks")
    print("       (close to 0 = no forgetting; negative = catastrophic forgetting)")
    print("  FWT: forward transfer — how much earlier tasks helped later ones")
    print("       (positive = generalization across tasks)")
    print()


def print_matrix(result: dict):
    """Print the full T×T accuracy matrix for one result."""
    matrix = result.get('matrix', [])
    if not matrix:
        return
    method = result.get('method', '?')
    print(f"  Accuracy matrix for {method}:")
    print(f"  (row = after training task k, col = test acc on task i)")
    print()
    T = len(matrix)
    # Header
    header = "    " + "".join(f"{f'task{i+1}':>9}" for i in range(T))
    print(header)
    print("  " + "─" * (4 + 9 * T))
    for k, row in enumerate(matrix):
        line = f"  k={k+1} | " + "".join(f"{a:>9.4f}" for a in row)
        print(line)
    print()


def main():
    p = argparse.ArgumentParser(description="Compare continual-learning runs")
    p.add_argument("results", nargs="+", help="One or more continual.py result JSON files")
    p.add_argument("--show-matrices", action="store_true",
                   help="Also print the full T×T accuracy matrix for each method")
    args = p.parse_args()

    results = [load_result(p) for p in args.results]
    print_table(results)

    if args.show_matrices:
        for r in results:
            print_matrix(r)


if __name__ == "__main__":
    main()
