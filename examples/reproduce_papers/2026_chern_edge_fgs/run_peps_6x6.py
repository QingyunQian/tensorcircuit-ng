"""
Reproduction of "Real-Time Dynamics in Two Dimensions with Tensor Network States
via Time-Dependent Variational Monte Carlo Method"
Link: https://arxiv.org/abs/2512.06768
Description:
Checkpointed 6 x 6 fermionic PEPS preparation and real-time corner quench.
The Gaussian solution is used only for independent validation after evolution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorcircuit as tc

if __package__:
    from .peps import FermionPEPS, Hofstadter
    from .run_peps import fgs_reference
    from .tvmc import MonteCarlo
else:
    from peps import FermionPEPS, Hofstadter
    from run_peps import fgs_reference
    from tvmc import MonteCarlo


def enlarge_bond(theta, source, target, noise=0.05):
    """Embed parity blocks exactly, then activate the new entries with seeded noise."""
    K = tc.backend
    positions = []
    for offset, small, large in zip(
        target.offsets, source.coordinates, target.coordinates
    ):
        lookup = {coordinate: index for index, coordinate in enumerate(large)}
        positions.extend(offset + lookup[coordinate] for coordinate in small)
    enlarged = K.scatter(
        K.zeros((target.nparams,), dtype="complex128"),
        K.reshape(K.convert_to_tensor(positions), (-1, 1)),
        theta,
    )
    return target.normalize(
        enlarged + noise * target.random_parameters(K.get_random_state(7109))
    )


def save_checkpoint(path, config, step, theta, states, key, history):
    """Atomically persist parameters, chains, RNG and pre-update diagnostics."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        config=json.dumps(config, sort_keys=True),
        step=step,
        theta=np.asarray(theta),
        states=np.asarray(states),
        key=np.asarray(key),
        history=history,
    )
    temporary.replace(path)


def main(placement=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "real_time"), default="real_time"
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--boundary-dim", type=int, default=None)
    parser.add_argument("--contraction-path", type=Path)
    parser.add_argument("--samples", type=int, default=10240)
    parser.add_argument("--contraction-batch-size", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--warmup-sweeps", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--chunk-steps", type=int, default=5)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--solver", choices=("sr", "dense", "minsr"), default="dense")
    parser.add_argument("--regulator", type=float, default=1e-5)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "6x6",
    )
    args = parser.parse_args()
    if (
        min(args.steps, args.chunk_steps, args.snapshot_every, args.warmup_sweeps) < 1
        or args.dt <= 0
    ):
        parser.error("Positive steps, snapshot interval, warmup and dt are required")
    imaginary = args.stage == "prepare"
    if not imaginary and args.initial_state is None:
        parser.error(
            "Real time requires a separately prepared --initial-state checkpoint"
        )
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    K = tc.backend
    peps = FermionPEPS(
        6, 6, args.bond_dim, args.boundary_dim, path_file=args.contraction_path
    )
    problem = Hofstadter(peps, 24)
    sampler = MonteCarlo(
        problem,
        args.samples,
        1,
        args.sweeps,
        args.solver,
        args.regulator,
        args.contraction_batch_size,
        args.cg_tolerance,
        args.cg_maxiter,
    )
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "output_dir",
            "initial_state",
            "contraction_path",
            "steps",
            "chunk_steps",
            "snapshot_every",
        )
    }
    config.update(
        rows=6, columns=6, particles=24, integrator="Euler VMC" if imaginary else "RK4"
    )
    source = Path(__file__).resolve().parent
    source_files = [
        source / name
        for name in (
            "peps.py",
            "tvmc.py",
            "sampling.py",
            "run_peps_6x6.py",
            "run_peps_6x6_optimized.py",
        )
    ]
    source_files.append(source.parents[1] / "peps_boundary_mps.py")
    config["source_sha256"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in source_files)
    ).hexdigest()
    if args.initial_state is not None:
        config["initial_state_sha256"] = hashlib.sha256(
            args.initial_state.read_bytes()
        ).hexdigest()
    if args.boundary_dim is None and args.contraction_path is not None:
        config["contraction_path_sha256"] = hashlib.sha256(
            args.contraction_path.read_bytes()
        ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.npz"
    started = time.perf_counter()
    if checkpoint.exists():
        saved = np.load(checkpoint)
        if json.loads(str(saved["config"])) != config:
            raise ValueError("Checkpoint physics or sampling configuration changed")
        theta, states, key = (
            K.convert_to_tensor(saved[name]) for name in ("theta", "states", "key")
        )
        if placement is not None:
            theta, states, key = placement(theta, states, key)
        history, step = saved["history"], int(saved["step"])
        if step > args.steps:
            parser.error("Requested endpoint precedes the saved checkpoint")
    else:
        key, draw = K.random_split(K.get_random_state(args.seed))
        theta = peps.random_parameters(draw)
        if args.initial_state is not None:
            initial = np.load(args.initial_state)
            initial_config = json.loads(str(initial["config"]))
            if not imaginary and initial_config["stage"] != "prepare":
                raise ValueError(
                    "Real time must start from a preparation checkpoint; resume through --output-dir"
                )
            if (initial_config["rows"], initial_config["columns"]) != (6, 6):
                raise ValueError("The initial checkpoint must be a 6 x 6 PEPS")
            theta = K.convert_to_tensor(initial["theta"])
            source_bond = initial_config["bond_dim"]
            if source_bond != args.bond_dim:
                if not imaginary or source_bond >= args.bond_dim:
                    raise ValueError(
                        "Bond enlargement is only supported during preparation"
                    )
                source = FermionPEPS(6, 6, source_bond, args.boundary_dim)
                theta = enlarge_bond(theta, source, peps)
        if theta.shape != (peps.nparams,):
            raise ValueError("The parameter array does not match the PEPS")
        key, draw = K.random_split(key)
        states = sampler.initial_chains(draw)
        if placement is not None:
            theta, states, key = placement(theta, states, key)
        if peps.boundary_dim is None:
            warmup = lambda p, s, k: sampler.advance(
                p, s, k, args.warmup_sweeps * problem.left.shape[0]
            )
        else:
            warmup = lambda p, s, k: sampler.sweep(p, s, k, args.warmup_sweeps)
        states, key, _ = K.jit(warmup)(theta, states, key)
        history, step = np.empty((0, 4 + 2 * peps.nsites)), 0
        save_checkpoint(checkpoint, config, step, theta, states, key, history)
        save_checkpoint(
            args.output_dir / "state-step-00000.npz",
            config,
            step,
            theta,
            states,
            key,
            history,
        )
    cache = {}
    while step < args.steps:
        count = min(
            args.chunk_steps,
            args.steps - step,
            args.snapshot_every - step % args.snapshot_every,
        )
        if count not in cache:
            cache[count] = K.jit(
                lambda p, s, k, n=count: sampler.trajectory(
                    p,
                    s,
                    k,
                    args.dt,
                    n,
                    -1.0 if imaginary else 0.0,
                    imaginary,
                    prepare_euler=imaginary,
                )
            )
        tick = time.perf_counter()
        theta, states, key, block = cache[count](theta, states, key)
        block = np.asarray(block)
        if not np.all(np.isfinite(block)) or not np.all(np.isfinite(np.asarray(theta))):
            raise FloatingPointError(
                "Nonfinite trajectory or unconverged CG; last valid checkpoint preserved"
            )
        history = np.concatenate((history, block))
        step += count
        save_checkpoint(checkpoint, config, step, theta, states, key, history)
        if step % args.snapshot_every == 0 or step == args.steps:
            save_checkpoint(
                args.output_dir / f"state-step-{step:05d}.npz",
                config,
                step,
                theta,
                states,
                key,
                history,
            )
        print(
            json.dumps(
                {
                    "step": step,
                    "time": step * args.dt,
                    "chunk_seconds_including_jit": time.perf_counter() - tick,
                    "pre_update_energy": float(block[-1, 0]),
                    "pre_update_variance": float(block[-1, 1]),
                    "sr_force_residual_squared": float(block[-1, 2]),
                }
            ),
            flush=True,
        )
    if not imaginary:
        _, final, _, _ = K.jit(sampler.estimate)(theta, states, key)
        diagnostics = np.concatenate((history, np.asarray(final)[None]))
        if not np.all(np.isfinite(diagnostics)):
            raise FloatingPointError("Nonfinite final diagnostics")
        times = np.arange(step + 1) * args.dt
        density, error = (
            diagnostics[:, 4 : 4 + peps.nsites],
            diagnostics[:, 4 + peps.nsites :],
        )
        exact, background, _ = fgs_reference(problem, times)
        np.testing.assert_allclose(density.sum(axis=1), 24, atol=1e-12)
        np.savez_compressed(
            args.output_dir / "peps_results.npz",
            times=times,
            density=density,
            density_stderr=error,
            exact_density=exact,
            background=background,
            diagnostics=diagnostics,
            final_theta=np.asarray(theta),
        )
    report = {
        "config": config,
        "completed_steps": step,
        "devices": len(theta.devices()),
        "invocation_seconds_including_warmup_jit_and_measurement": time.perf_counter()
        - started,
    }
    (args.output_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
