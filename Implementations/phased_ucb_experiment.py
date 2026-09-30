"""Empirical comparison for Exercise 7.5 (Phased UCB 2).

The script compares ordinary UCB(delta) with two batched variants:
  * phased_ucb1: the arm selected at a phase boundary is played for a
    pre-specified phase length (the previous-exercise style).
  * phased_ucb2: the selected arm is played until its own pull count is
    multiplied by alpha, exactly as in Exercise 7.5.

Run:
    python phased_ucb_experiment.py --horizon 10000 --runs 500 --alpha 2

The output figure contains mean cumulative pseudo-regret and error bars.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Bandit:
    """Bernoulli multi-armed bandit."""

    means: np.ndarray

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.means[arm])


def ucb_index(sums: np.ndarray, counts: np.ndarray, delta: float) -> np.ndarray:
    """UCB_i = empirical_mean_i + sqrt(2 log(1/delta) / T_i)."""
    empirical = sums / counts
    bonus = np.sqrt(2.0 * np.log(1.0 / delta) / counts)
    return empirical + bonus


def choose_and_observe(
    bandit: Bandit, arm: int, horizon: int, counts: np.ndarray,
    sums: np.ndarray, regret: np.ndarray, rng: np.random.Generator,
) -> int:
    """Pull arm once; return the next time index."""
    t = int(counts.sum())
    if t >= horizon:
        return t
    sums[arm] += bandit.sample(arm, rng)
    counts[arm] += 1
    regret[t] = bandit.means.max() - bandit.means[arm]
    return t + 1


def run_ucb(bandit: Bandit, horizon: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    k = len(bandit.means)
    counts, sums, regret = np.zeros(k, dtype=int), np.zeros(k), np.zeros(horizon)
    for arm in range(k):
        choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    while counts.sum() < horizon:
        arm = int(np.argmax(ucb_index(sums, counts, delta)))
        choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    return np.cumsum(regret)


def run_phased_ucb1(
    bandit: Bandit, horizon: int, delta: float, rng: np.random.Generator,
) -> np.ndarray:
    k = len(bandit.means)
    counts, sums, regret = np.zeros(k, dtype=int), np.zeros(k), np.zeros(horizon)
    for arm in range(k):
        choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    phase = 1
    while counts.sum() < horizon:
        arm = int(np.argmax(ucb_index(sums, counts, delta)))
        phase_length = 2 ** (phase-1)
        for _ in range(phase_length):
            if counts.sum() >= horizon:
                break
            choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
        phase += 1
    return np.cumsum(regret)


def run_phased_ucb2(
    bandit: Bandit, horizon: int, delta: float, alpha: float, rng: np.random.Generator,
) -> np.ndarray:
    """Exercise 7.5 algorithm: play selected arm until T_i >= alpha*T_i(start)."""
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1")
    k = len(bandit.means)
    alpha = 2
    counts, sums, regret = np.zeros(k, dtype=int), np.zeros(k), np.zeros(horizon)
    for arm in range(k):
        choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    while counts.sum() < horizon:
        arm = int(np.argmax(ucb_index(sums, counts, delta)))
        start_count = counts[arm]
        target = int(np.ceil(alpha * start_count))
        while counts[arm] < target and counts.sum() < horizon:
            choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    return np.cumsum(regret)


def run_phased_ucb3(
    bandit: Bandit, horizon: int, delta: float, alpha: float, rng: np.random.Generator,
) -> np.ndarray:
    """Exercise 7.5 algorithm: play selected arm until T_i >= alpha*T_i(start)."""
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1")
    k = len(bandit.means)
    counts, sums, regret = np.zeros(k, dtype=int), np.zeros(k), np.zeros(horizon)
    for arm in range(k):
        choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    while counts.sum() < horizon:
        arm = int(np.argmax(ucb_index(sums, counts, delta)))
        start_count = counts[arm]
        target = int(np.ceil(alpha * start_count))
        while counts[arm] < target and counts.sum() < horizon:
            choose_and_observe(bandit, arm, horizon, counts, sums, regret, rng)
    return np.cumsum(regret)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--means", type=float, nargs="+", default=[0.70, 0.50, 0.30])
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("phased_ucb_comparison.png"))
    args = parser.parse_args()

    bandit = Bandit(np.asarray(args.means, dtype=float))
    if not np.all((0 <= bandit.means) & (bandit.means <= 1)):
        raise ValueError("--means must be Bernoulli probabilities in [0, 1]")
    delta = args.horizon ** -2  # the choice used in the proof
    algorithms = {
        "UCB($\\delta$)": lambda g: run_ucb(bandit, args.horizon, delta, g),
        "Phased UCB 1": lambda g: run_phased_ucb1(bandit, args.horizon, delta, g),
        f"Phased UCB 2 ($\\alpha={2}$)": lambda g: run_phased_ucb2(
            bandit, args.horizon, delta, 2, g
        ),
        f"Phased UCB 3 ($\\alpha={args.alpha:g}$)": lambda g: run_phased_ucb3(
            bandit, args.horizon, delta, args.alpha, g
        ),
    }
    master_rng = np.random.default_rng(args.seed)
    x = np.arange(1, args.horizon + 1)
    plt.figure(figsize=(8, 5))
    for name, algorithm in algorithms.items():
        paths = np.vstack([algorithm(np.random.default_rng(master_rng.integers(2**63))) for _ in range(args.runs)])
        mean = paths.mean(axis=0)
        se = paths.std(axis=0, ddof=1) / np.sqrt(args.runs)
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=0.16)
        print(f"{name:28s} final regret = {mean[-1]:.3f} ± {1.96 * se[-1]:.3f} (95% CI)")
    plt.xlabel("round t")
    plt.ylabel("mean cumulative pseudo-regret")
    plt.title(f"Bernoulli means={args.means}, {args.runs} independent runs")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
