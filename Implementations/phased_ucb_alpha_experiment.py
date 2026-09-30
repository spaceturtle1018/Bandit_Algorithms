"""Show how Phased UCB 2 approaches UCB as alpha approaches one.

Example:
    python phased_ucb_experiment.py --horizon 20000 --runs 1000

The left panel plots mean cumulative pseudo-regret. The right panel is the
main result: final pseudo-regret against alpha, with a 95% confidence interval.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BernoulliBandit:
    means: np.ndarray

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.means[arm])


def indices(reward_sums: np.ndarray, pulls: np.ndarray, delta: float) -> np.ndarray:
    return reward_sums / pulls + np.sqrt(2 * np.log(1 / delta) / pulls)


def pull(bandit: BernoulliBandit, arm: int, horizon: int, pulls: np.ndarray,
         reward_sums: np.ndarray, regret: np.ndarray, rng: np.random.Generator) -> None:
    t = int(pulls.sum())
    if t == horizon:
        return
    reward_sums[arm] += bandit.sample(arm, rng)
    pulls[arm] += 1
    regret[t] = bandit.means.max() - bandit.means[arm]


def initialise(bandit: BernoulliBandit, horizon: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon < len(bandit.means):
        raise ValueError("horizon must be at least the number of arms")
    pulls = np.zeros(len(bandit.means), dtype=int)
    reward_sums, regret = np.zeros(len(bandit.means)), np.zeros(horizon)
    for arm in range(len(bandit.means)):
        pull(bandit, arm, horizon, pulls, reward_sums, regret, rng)
    return pulls, reward_sums, regret


def run_ucb(bandit: BernoulliBandit, horizon: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    pulls, reward_sums, regret = initialise(bandit, horizon, rng)
    while pulls.sum() < horizon:
        arm = int(np.argmax(indices(reward_sums, pulls, delta)))
        pull(bandit, arm, horizon, pulls, reward_sums, regret, rng)
    return np.cumsum(regret)


def run_phased_ucb2(bandit: BernoulliBandit, horizon: int, delta: float,
                    alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Play the selected arm until its count reaches alpha times its start count."""
    if alpha <= 1:
        raise ValueError("alpha must be strictly greater than 1")
    pulls, reward_sums, regret = initialise(bandit, horizon, rng)
    while pulls.sum() < horizon:
        arm = int(np.argmax(indices(reward_sums, pulls, delta)))
        target = int(np.ceil(alpha * pulls[arm]))
        while pulls[arm] < target and pulls.sum() < horizon:
            pull(bandit, arm, horizon, pulls, reward_sums, regret, rng)
    return np.cumsum(regret)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--means", type=float, nargs="+", default=[0.70, 0.50, 0.30])
    parser.add_argument("--horizon", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.01, 1.05, 1.10, 1.25, 1.50, 2.0, 3.0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("alpha_vs_regret.png"))
    args = parser.parse_args()
    alphas = np.asarray(args.alphas, dtype=float)
    means = np.asarray(args.means, dtype=float)
    if np.any(alphas <= 1):
        raise ValueError("all --alphas values must be greater than 1")
    if not np.all((0 <= means) & (means <= 1)):
        raise ValueError("--means must be probabilities in [0, 1]")

    bandit, delta = BernoulliBandit(means), args.horizon ** -2
    # The same seed is used for each method in one replication, reducing noise
    # when comparing different values of alpha.
    seeds = np.random.default_rng(args.seed).integers(0, 2**63 - 1, size=args.runs)
    ucb_paths = np.vstack([run_ucb(bandit, args.horizon, delta, np.random.default_rng(s)) for s in seeds])
    phased_paths = {
        alpha: np.vstack([run_phased_ucb2(bandit, args.horizon, delta, alpha, np.random.default_rng(s)) for s in seeds])
        for alpha in alphas
    }

    fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(12, 4.6))
    rounds, ucb_mean = np.arange(1, args.horizon + 1), ucb_paths.mean(axis=0)
    ax_curve.plot(rounds, ucb_mean, color="black", linewidth=2.4, label="UCB($\\delta$)")
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(alphas)))
    final_means, final_cis = [], []
    for alpha, colour in zip(alphas, colours):
        paths = phased_paths[alpha]
        mean = paths.mean(axis=0)
        ci = 1.96 * paths.std(axis=0, ddof=1)[-1] / np.sqrt(args.runs)
        ax_curve.plot(rounds, mean, color=colour, label=fr"$\alpha={alpha:g}$")
        final_means.append(mean[-1])
        final_cis.append(ci)
        print(f"alpha={alpha:>4g}: final regret = {mean[-1]:.3f} ± {ci:.3f}")

    ax_curve.set(xlabel="round $t$", ylabel="mean cumulative pseudo-regret", title="Cumulative regret")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=8)
    ax_final.errorbar(alphas, final_means, yerr=final_cis, fmt="o-", capsize=3, color="#3b528b")
    ax_final.axhline(ucb_mean[-1], color="black", linestyle="--", label="UCB($\\delta$)")
    ax_final.set(xlabel=r"phase multiplier $\alpha$", ylabel="final cumulative pseudo-regret", title=r"Smaller $\alpha$ approaches UCB")
    ax_final.grid(alpha=0.25)
    ax_final.legend()
    fig.suptitle(f"Bernoulli means={args.means}; {args.runs} replications", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"UCB(delta): final regret = {ucb_mean[-1]:.3f}")
    print(f"Saved plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
