"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Description:
This example reproduces Figures 1(d), 1(e), and 3 with TensorCircuit-NG.
The default L=8 demo is a finite-size demonstration. The L=16 preset is costly.
Use --plot-only to plot the L=16 results from outputs/summary.npz.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from accelerated import JaxKernels
from plotting import aggregate, plot
from simulation import sample

import tensorcircuit as tc

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("demo", "paper"), default="demo")
    parser.add_argument(
        "--engine",
        choices=("jax", "numpy"),
        default="jax",
        help="Entropy/circuit engine; spectral preparation always uses NumPy",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path relative to the example, or an absolute path",
    )
    args = parser.parse_args()
    output = args.output or Path(
        "outputs" if args.plot_only else f"outputs/{args.preset}/{args.engine}"
    )
    output = HERE / output
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with np.load(output / "summary.npz", allow_pickle=False) as summary:
            provenance = json.loads((output / "provenance.json").read_text())
            plot(summary, output, provenance["plot_label"])
        return

    # Preserve CPU eigensolvers, Schur phases and longdouble phase reduction.
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")
    config = json.loads((HERE / f"{args.preset}.json").read_text())
    config["engine"] = args.engine
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains a different configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    kernels = JaxKernels(config["L"]) if args.engine == "jax" else None
    for group in ("hamiltonian", "rqc", "baee"):
        for number in range(config["samples"]):
            sample(config, group, number, output, kernels)
    summary = aggregate(output, config)
    np.savez_compressed(output / "summary.npz", **summary)
    label = (
        "Paper-scale reproduction"
        if args.preset == "paper"
        else "Small-system demonstration"
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "plot_label": label,
                "tensorcircuit_version": tc.__version__,
                "config": config,
                "statistical_unit": "independent disorder sample",
            },
            indent=2,
        )
        + "\n"
    )
    plot(summary, output, label)
    print(f"Saved {output / 'result.png'}")


if __name__ == "__main__":
    main()
