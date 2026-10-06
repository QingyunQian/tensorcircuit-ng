"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Aggregate independent samples and plot Figures 1(d), 1(e), and 3.
"""

import hashlib
import json
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

GROUP_KEYS = {
    "hamiltonian": ["initial_hcee"]
    + [f"sat_{name}" for name in ("thermal", "AL", "free", "MBL", "Floquet")],
    "rqc": ["initial_hcee", "initial_baee"]
    + [f"rqc_sat_{name}" for name in ("thermal", "A", "B", "C", "D")],
    "baee": ["baee_hcee", "baee_mean"],
}
COLORS = ["#e31a1c", "#33a02c", "#1f78b4", "#fdbf6f", "#ff7f00", "#6a3d9a"]


def aggregate(directory, config):
    """Use disorder samples as statistical units, including paired differences."""
    summary = {"L": config["L"], "samples": config["samples"]}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for group, keys in GROUP_KEYS.items():
        records = []
        for number in range(config["samples"]):
            with np.load(Path(directory) / f"{group}_{number:04d}.npz") as data:
                if str(data["config_sha256"]) != digest:
                    raise ValueError("Sample configuration differs from this run")
                records.append(dict(data))
        for key in ("tau", "baee_times"):
            if key in records[0]:
                summary[key] = records[0][key]
        for key in keys:
            values = np.array([record[key] for record in records])
            save_statistics(summary, f"{group}_{key}", values)
            if key.startswith(("sat_", "rqc_sat_")) or key == "initial_baee":
                initial = np.array([record["initial_hcee"] for record in records])
                save_statistics(summary, f"{group}_{key}_growth", values - initial)
        if group == "baee":
            save_statistics(
                summary,
                "baee_reservoir",
                np.array(
                    [record["baee_mean"] - record["baee_hcee"] for record in records]
                ),
            )
    for key, value in summary.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite summary: {key}")
    return summary


def save_statistics(summary, key, values):
    """The configured presets use at least two samples for a defined SEM."""
    summary[key] = values.mean(axis=0)
    summary[key + "_sem"] = values.std(axis=0, ddof=1) / np.sqrt(len(values))


def haar_half_filling(length):
    """Mean complex-Haar half-chain entropy in the half-filled sector, in bits."""
    dimension = comb(length, length // 2)
    answer = 0.0
    for charge in range(length // 2 + 1):
        block_size = comb(length // 2, charge)
        weight = block_size**2 / dimension
        answer += weight * (
            digamma(dimension + 1)
            - digamma(block_size + 1)
            - (block_size - 1) / (2 * block_size)
        )
    return float(answer / np.log(2))


def plot(summary, output, label):
    """Plot the three target panels with one-SEM bars and the Haar reference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.91, wspace=0.25)

    def line(ax, x, key, name, color, xerror=None):
        ax.errorbar(
            x,
            summary[key],
            yerr=summary[key + "_sem"],
            xerr=xerror,
            label=name,
            color=color,
            linewidth=1.5,
            elinewidth=0.8,
            capsize=2,
            marker=".",
            markersize=3,
        )

    for name, display, color in zip(
        ("thermal", "AL", "free", "MBL", "Floquet"),
        ("thermal", "AL", "free fermion", "Hamiltonian MBL", "Floquet MBL"),
        COLORS,
    ):
        line(
            axes[0],
            summary["hamiltonian_initial_hcee"],
            f"hamiltonian_sat_{name}_growth",
            display,
            color,
            summary["hamiltonian_initial_hcee_sem"],
        )
    circuit_labels = (
        r"RQC thermal $\alpha=\pi/2,\ \beta=\pi$",
        r"RQC(A) $\alpha=0,\ \beta=\pi$",
        r"RQC(B) $\alpha=\pi/2,\ \beta=0$",
        r"RQC(C) $\alpha=\pi,\ \beta=0$",
        r"RQC(D) $\alpha=\pi,\ \beta=\pi/2$",
        "random SWAP circuit",
    )
    for name, display, color in zip(
        ("thermal", "A", "B", "C", "D", "SWAP"), circuit_labels, COLORS
    ):
        key = "initial_baee" if name == "SWAP" else f"rqc_sat_{name}"
        line(
            axes[1],
            summary["rqc_initial_hcee"],
            f"rqc_{key}_growth",
            display,
            color,
            summary["rqc_initial_hcee_sem"],
        )
    for key, name, color in zip(
        ("baee_hcee", "baee_mean", "reservoir"),
        (r"$S$", r"$\bar{S}$", r"$\bar{S}-S$"),
        (COLORS[2], COLORS[4], COLORS[1]),
    ):
        line(axes[2], summary["baee_times"], f"baee_{key}", name, color)
    length = int(summary["L"])
    haar = haar_half_filling(length)
    for index, (ax, title) in enumerate(
        zip(axes, ("Fig. 1(d)", "Fig. 1(e)", "Fig. 3"))
    ):
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, length / 2)
        ax.set_ylabel(r"$\Delta S$" if index < 2 else "EE", fontsize=15)
        ax.set_xlabel(
            r"$S_{\mathrm{initial}}$" if index < 2 else r"$\tau$", fontsize=15
        )
        ax.axhline(haar, color="black", linestyle="--", linewidth=1.5, zorder=0)
        if index < 2:
            ax.set_xlim(0, length / 2)
            ax.axvline(haar, color="black", linestyle="--", linewidth=1.5, zorder=0)
        else:
            ax.set_xlim(0, 50)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(
            fontsize=9 if index < 2 else 13,
            loc="upper right" if index < 2 else "center right",
        )
    fig.text(
        0.5,
        0.035,
        f"{label}  |  L={length}, "
        f"{int(summary['samples'])} disorder samples per group  |  "
        "Error bars: one SEM; dashed lines: fixed-charge Haar mean",
        ha="center",
        fontsize=10,
    )
    fig.savefig(Path(output) / "result.png", dpi=150)
    plt.close(fig)
