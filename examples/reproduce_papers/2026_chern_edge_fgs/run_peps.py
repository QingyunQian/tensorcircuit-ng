"""
Reproduction of "Real-Time Dynamics in Two Dimensions with Tensor Network States
via Time-Dependent Variational Monte Carlo Method"
Link: https://arxiv.org/abs/2512.06768
Description:
This script studies the Figure 2(b,c) quench with number-projected fermionic
PEPS-tVMC and an independent TensorCircuit-NG FGS reference.
"""

import argparse
import json
import math
import importlib.metadata
from pathlib import Path
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tencirpauli as tcp
import tensorcircuit as tc

if __package__:
    from .main import correlation_at_time, fixed_number_alpha
    from .peps import FermionPEPS, Hofstadter
    from .tvmc import MonteCarlo, sr_cg_solve, sr_solve
else:
    from main import correlation_at_time, fixed_number_alpha
    from peps import FermionPEPS, Hofstadter
    from tvmc import MonteCarlo, sr_cg_solve, sr_solve


def peak_memory_mib():
    """Normalize the platform-dependent peak RSS units."""
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def fgs_reference(problem, times):
    """Reuse the validated FGS example for the identical rectangular Hamiltonian."""
    K = tc.backend
    h = problem.one_body()
    initial = problem.one_body(-1.0)

    def nambu(matrix):
        zero = K.zeros(matrix.shape, dtype="complex128")
        return (
            K.concat(
                [
                    K.concat([matrix, zero], axis=1),
                    K.concat([zero, -K.transpose(matrix)], axis=1),
                ],
                axis=0,
            )
            / 2
        )

    hf, initial_hf = nambu(h), nambu(initial)
    alpha = fixed_number_alpha(initial_hf, problem.particles)
    reference = tc.FGSSimulator(
        problem.nsites, alpha=fixed_number_alpha(hf, problem.particles)
    ).get_cmatrix()
    evolve = K.jit(K.vmap(lambda t: correlation_at_time(alpha, hf, t)))
    correlations = evolve(K.convert_to_tensor(times))
    densities = K.real(
        K.einsum("tii->ti", correlations[:, problem.nsites :, problem.nsites :])
    )
    background = K.real(
        K.einsum("ii->i", reference[problem.nsites :, problem.nsites :])
    )
    ground_energy = K.sum(K.eigh(initial)[0][: problem.particles])
    return tuple(np.asarray(K.numpy(x)) for x in (densities, background, ground_energy))


def run_trajectory(
    sampler,
    theta,
    chains,
    key,
    dt,
    steps,
    potential,
    imaginary,
    prepare_euler=False,
):
    """Stage the entire fixed-step trajectory and transfer diagnostics once."""
    K = tc.backend
    run = K.jit(
        lambda p, s, k: sampler.trajectory(
            p, s, k, dt, steps, potential, imaginary, prepare_euler
        )
    )
    started = time.perf_counter()
    theta, chains, key, history = run(theta, chains, key)
    history = np.asarray(K.numpy(history))
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(K.numpy(theta))):
        raise FloatingPointError(
            "Non-finite PEPS trajectory; inspect contraction and SR conditioning."
        )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "stage": "ground_state" if imaginary else "real_time",
                "steps": steps,
                "energy": float(history[-1, 0]),
                "variance": float(history[-1, 1]),
                "sr_residual": float(history[-1, 2]),
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )
    return theta, chains, key, history, elapsed


def plot_result(
    times,
    density,
    errors,
    exact,
    background,
    peps,
    output,
    filename="peps_result.png",
):
    """Compare measured densities with FGS using the Figure 2 color conventions."""
    fig = plt.figure(figsize=(12.4, 8.0), layout="constrained")
    layout = fig.add_gridspec(3, 4, height_ratios=(1, 1, 1.15))
    selected = np.linspace(0, len(times) - 1, 4, dtype=int)
    delta, reference_delta = density - background, exact - background
    coordinates = (
        (0, peps.rows - 1),
        ((peps.columns - 1) // 2, peps.rows - 1),
        (peps.columns - 1, peps.rows - 1),
        (peps.columns - 1, peps.rows // 2),
    )
    sites = tuple(y * peps.columns + x for x, y in coordinates)
    maps = np.concatenate([delta[selected], reference_delta[selected]])
    below, above = np.any(maps < 0), np.any(maps > 0.04)
    extend = (
        "both" if below and above else "min" if below else "max" if above else "neither"
    )
    for row, values in enumerate((reference_delta, delta)):
        for column, index in enumerate(selected):
            ax = fig.add_subplot(layout[row, column])
            mesh = ax.pcolormesh(
                np.arange(peps.columns + 1),
                np.arange(peps.rows + 1),
                values[index].reshape(peps.rows, peps.columns),
                cmap="viridis_r",
                vmin=0.0,
                vmax=0.04,
                edgecolors=(0, 0, 0, 0.15),
                linewidth=0.3,
            )
            ax.set_title(f"{'FGS' if row == 0 else 'PEPS-tVMC'}  t={times[index]:g}")
            ax.set(
                aspect="equal", xlabel="x", xlim=(0, peps.columns), ylim=(0, peps.rows)
            )
            ax.set_xticks(range(0, peps.columns + 1, max(1, peps.columns // 6)))
            ax.set_yticks(range(0, peps.rows + 1, max(1, peps.rows // 6)))
            if column == 0:
                ax.set_ylabel("y")
            if row == 0:
                x, y = coordinates[column]
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    "ABCD"[column],
                    color="red",
                    ha="center",
                    va="center",
                )
    fig.colorbar(
        mesh,
        ax=fig.axes[:8],
        extend=extend,
        label=r"$\langle n_i(t)\rangle-\langle n_i\rangle_{\rm gs}$",
    )
    lower = np.minimum(
        np.min(exact[:, sites], axis=0),
        np.min(density[:, sites] - errors[:, sites], axis=0),
    )
    upper = np.maximum(
        np.max(exact[:, sites], axis=0),
        np.max(density[:, sites] + errors[:, sites], axis=0),
    )
    span = max(0.2, float(np.max(upper - lower)) + 0.02)
    stride = max(1, (len(times) - 1) // 60)
    for column, site in enumerate(sites):
        ax = fig.add_subplot(layout[2, column])
        ax.plot(
            times,
            exact[:, site],
            color="tab:orange",
            linewidth=1.2,
            zorder=3,
            label="FGS exact",
        )
        ax.errorbar(
            times[::stride],
            density[::stride, site],
            yerr=errors[::stride, site],
            fmt=".",
            markersize=3,
            color="tab:blue",
            linewidth=0.6,
            zorder=2,
            label="PEPS-tVMC",
        )
        x, y = coordinates[column]
        ax.set(
            title=f"Site {'ABCD'[column]} ({x}, {y})",
            xlabel="t",
            ylabel=r"$\langle n_i\rangle$",
            xlim=(times[0], times[-1]),
        )
        center = (lower[column] + upper[column]) / 2
        ax.set_ylim(center - span / 2, center + span / 2)
        if column == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Fermionic PEPS-tVMC · {peps.rows} × {peps.columns}, D={peps.bond_dim}\n"
        "Paper map scale: 0–0.04; colorbar extensions mark values outside this range",
        fontsize=12,
    )
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def profile(problem, sampler, theta, chains, key, output):
    """Measure contraction, score, local energy, and one MC proposal separately."""
    K = tc.backend
    results = {}
    functions = {
        "amplitude": (K.jit(problem.peps.batch_amplitude), (theta, chains)),
        "scores": (K.jit(problem.peps.batch_scores), (theta, chains)),
        "local_energy": (K.jit(problem.local_batch), (theta, chains)),
        "mc_proposal": (
            K.jit(lambda p, s, k: sampler.advance(p, s, k, 1)),
            (theta, chains, key),
        ),
    }
    for name, (function, arguments) in functions.items():
        elapsed = []
        for _ in range(2):
            start = time.perf_counter()
            values = function(*arguments)
            leaves = K.tree_map(lambda x: np.asarray(K.numpy(x)), values)
            if not all(np.all(np.isfinite(x)) for x in K.tree_flatten(leaves)[0]):
                raise FloatingPointError(f"Non-finite {name} output.")
            elapsed.append(time.perf_counter() - start)
        results[name] = {"first_seconds": elapsed[0], "steady_seconds": elapsed[1]}
        print(json.dumps({name: results[name]}), flush=True)
    report = {
        "rows": problem.peps.rows,
        "columns": problem.peps.columns,
        "D": problem.peps.bond_dim,
        "boundary_dim": problem.peps.boundary_dim,
        "parameters": problem.peps.nparams,
        "batch_size": sampler.chains,
        "timings": results,
        "max_rss_mib": peak_memory_mib(),
        "tensorcircuit": tc.__version__,
        "tencirpauli": tcp.__version__,
    }
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def profile_scaling(args):
    """
    Reproducible solver-only storage benchmark; run each case in a fresh process.

    Synthetic scores isolate linear algebra from contraction/sampling. Report
    compiler buffers and device allocator peak separately from host RSS.
    """
    K = tc.backend
    peps = FermionPEPS(args.rows, args.columns, args.bond_dim, boundary_dim=8)
    count, parameters = args.chains * args.draws, peps.nparams
    if args.solver not in ("sr", "dense"):
        raise ValueError("The scaling comparison uses sr or dense.")
    if args.solver == "dense" and parameters > 6000:
        raise ValueError(
            "Dense benchmark limited to P <= 6000; larger cases are estimates only."
        )
    key1, key2 = K.random_split(K.get_random_state(args.seed))
    keys = (*K.random_split(key1), *K.random_split(key2))
    scores = (
        K.stateful_randn(keys[0], (count, parameters))
        + 1j * K.stateful_randn(keys[1], (count, parameters))
    ) / math.sqrt(2)
    energies = K.stateful_randn(keys[2], (count,)) + 1j * K.stateful_randn(
        keys[3], (count,)
    )
    weights = K.ones((count,), dtype="float64") / count
    if args.solver == "sr":
        function = lambda o, e, w: sr_cg_solve(
            o, e, w, 1e-3, args.cg_tolerance, args.cg_maxiter
        )
    else:
        # No gauge generators in synthetic data: a zero-width thin basis.
        function = lambda o, e, w: sr_solve(
            o,
            e,
            w,
            regulator=1e-3,
            gauge_basis=K.zeros((parameters, 0), dtype="complex128"),
        )
    report, result = timed_executable(function, (scores, energies, weights))
    report.update(
        {
            "rows": args.rows,
            "columns": args.columns,
            "D": args.bond_dim,
            "N": count,
            "P": parameters,
            "solver": args.solver,
            "regulator": 1e-3,
            "dtype": "complex128",
            "seed": args.seed,
            "score_bytes": 16 * count * parameters,
            "single_dense_metric_bytes": 16 * parameters**2,
            "synthetic_scores": True,
            "force_residual_squared": float(result[2]),
        }
    )
    if args.solver == "sr":
        report.update(
            {"linear_residual": float(result[3]), "iterations": int(result[4])}
        )
        np.testing.assert_allclose(result[3], 0, atol=args.cg_tolerance * 10)
    return report


def timed_executable(function, arguments):
    """JAX-only profiling boundary; numerical kernels still use tc.backend."""
    start = time.perf_counter()
    lowered = tc.backend.jit(function).lower(*arguments)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    executable = lowered.compile()
    compilation = time.perf_counter() - start
    durations = []
    for _ in range(4):
        start = time.perf_counter()
        result = executable(*arguments)
        for value in result:
            value.block_until_ready()
        durations.append(time.perf_counter() - start)
    memory = executable.memory_analysis()
    stats = result[0].device.memory_stats()
    return {
        "lower_seconds": lowering,
        "compile_seconds": compilation,
        "first_seconds": durations[0],
        "steady_seconds": float(np.median(durations[1:])),
        "compiler_argument_bytes": memory.argument_size_in_bytes,
        "compiler_output_bytes": memory.output_size_in_bytes,
        "compiler_temp_bytes": memory.temp_size_in_bytes,
        "device_peak_bytes": stats["peak_bytes_in_use"] if stats else None,
        "host_peak_mib": peak_memory_mib(),
        "device": str(result[0].device),
        "jax_version": importlib.metadata.version("jax"),
        "jaxlib_version": importlib.metadata.version("jaxlib"),
    }, result


def profile_exact_path(args):
    """Plan once and time the frozen path's amplitude and complex score."""
    K = tc.backend
    peps = FermionPEPS(
        args.rows,
        args.columns,
        args.bond_dim,
        exact_optimizer=args.exact_optimizer,
        path_file=args.contraction_path,
    )
    topology = peps.exact_path_record["topology"]
    current = [list(site) for site in topology["inputs"]]
    flops, write, largest = 0, 0, 1
    for a, b, _ in peps.exact_steps:
        common = set(current[a]) & set(current[b])
        remaining = [ix for ix in current[a] + current[b] if ix not in common]
        size = args.bond_dim ** len(remaining)
        flops += (2 if common else 1) * args.bond_dim ** len(
            set(current[a] + current[b])
        )
        write += size
        largest = max(largest, size)
        current = [site for i, site in enumerate(current) if i not in (a, b)] + [
            remaining
        ]
    report = {
        "rows": args.rows,
        "columns": args.columns,
        "D": args.bond_dim,
        "optimizer": peps.exact_path_record["optimizer"],
        "path_search_or_load_seconds": peps.exact_path_seconds,
        "ntensors": peps.nsites,
        "nindices": len(peps.bonds),
        "log10_flops": math.log10(flops),
        "log2_max_intermediate": math.log2(largest),
        "log2_write": math.log2(write),
        "sliced_indices": 0,
        "slice_tasks": 1,
        "slicing_seconds": 0,
        "path": peps.exact_path_record,
        "opt_einsum_version": importlib.metadata.version("opt_einsum"),
    }
    if str(report["optimizer"]).startswith("omeco"):
        report["omeco_version"] = importlib.metadata.version("omeco")
    if largest > 2**26:
        report["execution_skipped"] = (
            "Forward intermediate exceeds 2**26 complex elements."
        )
        return report
    theta = peps.random_parameters(K.get_random_state(args.seed))
    occupation = K.cast(K.arange(peps.nsites) < 2 * peps.nsites // 3, "int64")
    timing, _ = timed_executable(peps.scores, (theta, occupation))
    report["amplitude_and_score"] = timing
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--boundary-dim", type=int)
    parser.add_argument("--chains", type=int, default=256)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--contraction-batch-size", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--prepare-steps", type=int, default=400)
    parser.add_argument("--prepare-dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--solver", choices=("sr", "minsr", "dense"), default="sr")
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=512)
    parser.add_argument("--exact-optimizer", default="omeco-8-48")
    parser.add_argument("--contraction-path", type=Path)
    parser.add_argument("--scaling-profile", action="store_true")
    parser.add_argument("--path-profile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    args = parser.parse_args()
    if args.dt <= 0 or args.time <= 0 or args.prepare_steps < 1 or args.prepare_dt <= 0:
        parser.error("Positive dt, time, and preparation steps are required.")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    K = tc.backend
    if args.scaling_profile or args.path_profile:
        report = (
            profile_scaling(args) if args.scaling_profile else profile_exact_path(args)
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "scaling_profile.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    peps = FermionPEPS(
        args.rows,
        args.columns,
        args.bond_dim,
        args.boundary_dim,
        args.exact_optimizer,
        args.contraction_path,
    )
    problem = Hofstadter(peps, 2 * peps.nsites // 3)
    sampler = MonteCarlo(
        problem,
        args.chains,
        args.draws,
        args.sweeps,
        args.solver,
        contraction_batch_size=args.contraction_batch_size,
        cg_tolerance=args.cg_tolerance,
        cg_maxiter=args.cg_maxiter,
    )
    key, draw = K.random_split(K.get_random_state(args.seed))
    theta = peps.random_parameters(draw)
    key, draw = K.random_split(key)
    chains = sampler.initial_chains(draw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.profile:
        print(
            json.dumps(
                profile(problem, sampler, theta, chains, key, args.output_dir), indent=2
            )
        )
        return
    steps = round(args.time / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, args.time):
        parser.error("time must be an integer multiple of dt")
    times = np.arange(steps + 1) * args.dt
    exact, background, ground_energy = fgs_reference(problem, times)
    if args.initial_state is not None:
        saved = np.load(args.initial_state)
        theta = K.convert_to_tensor(saved["initial_theta"])
        if (
            theta.shape != (peps.nparams,)
            or int(saved["rows"]) != args.rows
            or int(saved["columns"]) != args.columns
            or int(saved["bond_dim"]) != args.bond_dim
        ):
            parser.error(
                "Initial PEPS does not match the chosen lattice and bond dimension."
            )
    thermalize = lambda p, s, k: (
        sampler.advance(p, s, k, 512)
        if peps.boundary_dim is None
        else sampler.sweep(p, s, k, 32)
    )
    chains, key, _ = K.jit(thermalize)(theta, chains, key)
    if args.initial_state is None:
        preparation_config = {
            "dt": args.prepare_dt,
            "steps": args.prepare_steps,
            "chains": args.chains,
            "draws": args.draws,
            "sweeps": args.sweeps,
            "seed": args.seed,
            "solver": args.solver,
            "regulator": 1e-4,
            "cg_tolerance": args.cg_tolerance,
            "cg_maxiter": args.cg_maxiter,
        }
        preparation = MonteCarlo(
            problem,
            args.chains,
            args.draws,
            args.sweeps,
            args.solver,
            regulator=1e-4,
            contraction_batch_size=args.contraction_batch_size,
            cg_tolerance=args.cg_tolerance,
            cg_maxiter=args.cg_maxiter,
        )
        theta, chains, key, prep_history, prep_seconds = run_trajectory(
            preparation,
            theta,
            chains,
            key,
            args.prepare_dt,
            args.prepare_steps,
            -1.0,
            True,
        )
    else:
        prep_history = saved["preparation"]
        preparation_config = json.loads(str(saved["preparation_config"]))
        prep_seconds = 0.0
    initial_theta = np.asarray(K.numpy(theta))
    np.savez_compressed(
        args.output_dir / "peps_initial_state.npz",
        initial_theta=initial_theta,
        preparation=prep_history,
        preparation_config=json.dumps(preparation_config),
        rows=args.rows,
        columns=args.columns,
        bond_dim=args.bond_dim,
    )
    theta, chains, key, history, evolve_seconds = run_trajectory(
        sampler, theta, chains, key, args.dt, steps, 0.0, False
    )
    _, final, _, _ = K.jit(sampler.estimate)(theta, chains, key)
    history = np.concatenate([history, np.asarray(K.numpy(final))[None]])
    density, error = history[:, 4 : 4 + peps.nsites], history[:, 4 + peps.nsites :]
    report = {
        "rows": args.rows,
        "columns": args.columns,
        "particles": problem.particles,
        "D": args.bond_dim,
        "boundary_dim": args.boundary_dim,
        "contraction_batch_size": args.contraction_batch_size,
        "sampling": "random_bonds" if args.boundary_dim is None else "sequential",
        "dt": args.dt,
        "time": args.time,
        "seed": args.seed,
        "samples_per_stage": args.chains * args.draws,
        "chains": args.chains,
        "draws": args.draws,
        "sweeps": args.sweeps,
        "solver": args.solver,
        "regulator": sampler.regulator,
        "cg_tolerance": args.cg_tolerance,
        "cg_maxiter": args.cg_maxiter,
        "exact_contraction_path": (
            peps.exact_path_record if peps.boundary_dim is None else None
        ),
        "preparation_config": preparation_config,
        "ground_state_exact_energy": float(ground_energy),
        "last_preparation_energy": float(prep_history[-1, 0]),
        "last_preparation_variance": float(prep_history[-1, 1]),
        "max_sampled_density_error_vs_fgs": float(np.max(np.abs(density - exact))),
        "max_sampled_energy_drift": float(
            np.max(np.abs(history[:, 0] - history[0, 0]))
        ),
        "max_sr_residual": float(np.max(history[:, 2])),
        "preparation_seconds_including_jit": prep_seconds,
        "evolution_seconds_including_jit": evolve_seconds,
        "max_rss_mib": peak_memory_mib(),
        "tensorcircuit": tc.__version__,
        "tencirpauli": tcp.__version__,
    }
    np.savez_compressed(
        args.output_dir / "peps_results.npz",
        times=times,
        density=density,
        density_stderr=error,
        exact_density=exact,
        background=background,
        diagnostics=history,
        preparation=prep_history,
        initial_theta=initial_theta,
        final_theta=np.asarray(K.numpy(theta)),
    )
    (args.output_dir / "peps_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    plot_result(times, density, error, exact, background, peps, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
