"""Plot independently measured 6 x 6 PEPS densities against the exact solution."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_measurements(trajectory, measurements, output_dir):
    """Use raw PEPS importance estimates and one common, fixed density color scale."""
    background = np.load(trajectory)["background"]
    records = [json.loads(path.read_text()) for path in measurements]
    arrays = [np.load(path.with_suffix(".npz")) for path in measurements]
    times = np.asarray([record["time"] for record in records])
    if background.shape != (36,) or not np.all(np.diff(times) > 0):
        raise ValueError("Provide a 6 x 6 trajectory and measurements in time order")
    density = np.stack([data["density"] for data in arrays])
    exact = np.stack([data["exact_density"] for data in arrays])
    stderr = np.stack([data["density_stderr"] for data in arrays])
    np.testing.assert_allclose(density.sum(axis=1), 24.0, atol=1e-10)
    np.testing.assert_allclose(exact.sum(axis=1), 24.0, atol=1e-10)
    difference = density - exact
    error_limit = max(0.004, float(np.max(np.abs(difference))))
    fig, axes = plt.subplots(3, len(times), figsize=(10.2, 7.5), squeeze=False)
    fig.subplots_adjust(
        left=0.10, right=0.86, bottom=0.07, top=0.89, hspace=0.18, wspace=0.12
    )
    for column, physical_time in enumerate(times):
        for row, values in enumerate(
            (exact - background, density - background, difference)
        ):
            ax = axes[row, column]
            image = ax.imshow(
                values[column].reshape(6, 6),
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r" if row == 2 else "viridis_r",
                vmin=-error_limit if row == 2 else 0.0,
                vmax=error_limit if row == 2 else 0.04,
            )
            ax.set_xticks([0, 5])
            ax.set_yticks([0, 5])
            ax.tick_params(labelsize=8, length=2)
            if row == 0:
                ax.set_title(f"t = {physical_time:g}", fontsize=12)
            if row == 2:
                ax.set_xlabel(
                    f"RMSE {np.sqrt(np.mean(difference[column]**2)):.4f}", fontsize=9
                )
            if column == 0:
                ax.set_ylabel(
                    ("Exact (FGS)", "PEPS-tVMC", "PEPS − exact")[row], fontsize=11
                )
        if column == 0:
            density_image = axes[0, 0].images[0]
            error_image = image
    fig.colorbar(
        density_image,
        cax=fig.add_axes([0.885, 0.38, 0.018, 0.49]),
        label="Density excess δn",
        extend="both",
    )
    fig.colorbar(
        error_image, cax=fig.add_axes([0.885, 0.09, 0.018, 0.20]), label="Density error"
    )
    fig.suptitle(
        "6 × 6 corner quench · 24 fermions · D = 4\nIndependent PEPS measurements; shared 0–0.04 color scale",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "peps_6x6.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        output_dir / "comparison.npz",
        times=times,
        density=density,
        exact_density=exact,
        density_stderr=stderr,
        background=background,
        density_difference=difference,
    )
    report = {
        "measurements": records,
        "density_rmse_all_snapshots": float(np.sqrt(np.mean(difference**2))),
        "density_max_error_all_snapshots": float(np.max(np.abs(difference))),
        "display": "Identical viridis_r scale [0, 0.04]; signed arrays are retained without clipping.",
        "uncertainty": "Jackknife errors cover measurement sampling, not accumulated evolution or ansatz bias.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "6x6",
    )
    args = parser.parse_args()
    plot_measurements(args.trajectory, args.measurements, args.output_dir)


if __name__ == "__main__":
    main()
