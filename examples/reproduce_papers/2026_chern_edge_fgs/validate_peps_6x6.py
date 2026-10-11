"""
Estimate PEPS fidelity independently with exact Gaussian Born samples.

The Gaussian state is a validation proposal only, never a training target.
PEPS amplitudes use exact single-layer contractions, without Fock enumeration.
Reported uncertainties cover sampling, not ansatz or time-step bias.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorcircuit as tc

if __package__:
    from .main import fixed_number_alpha
    from .peps import FermionPEPS, Hofstadter
    from .validate_peps import sector_reference
else:
    from main import fixed_number_alpha
    from peps import FermionPEPS, Hofstadter
    from validate_peps import sector_reference


def gaussian_reference(problem, physical_time):
    K = tc.backend
    initial = problem.one_body(-1.0)
    zero = K.zeros(initial.shape, dtype="complex128")
    nambu = (
        K.concat(
            [
                K.concat([initial, zero], axis=1),
                K.concat([zero, -K.transpose(initial)], axis=1),
            ],
            axis=0,
        )
        / 2
    )
    alpha = fixed_number_alpha(nambu, problem.particles)
    orbitals = K.eigh(initial)[1][:, : problem.particles]
    h = problem.one_body()
    orbitals = K.expm(-1j * physical_time * h) @ orbitals
    nambu = (
        K.concat(
            [K.concat([h, zero], axis=1), K.concat([zero, -K.transpose(h)], axis=1)],
            axis=0,
        )
        / 2
    )
    state = tc.FGSSimulator(problem.nsites, alpha=alpha)
    state.evol_hamiltonian(2 * physical_time * nambu)
    return state.alpha, orbitals


def born_draw_fgs(alpha, randoms):
    K = tc.backend
    n = randoms.shape[0]

    def measure(carry, i):
        current, occupation = carry
        state = tc.FGSSimulator(n, alpha=current)
        value = state.cond_measure(i, randoms[i])
        occupation = K.scatter(occupation, K.reshape(i, (1, 1)), K.reshape(value, (1,)))
        return state.alpha, occupation

    _, occupation = K.scan(
        measure, K.arange(n), (alpha, K.zeros((n,), dtype="float64"))
    )
    return K.cast(occupation, "int32")


def born_draw(correlation, randoms):
    """Exact number-conserving Gaussian sampling by Wick's conditional update.

    For unmeasured j,k, observing n_i=1 subtracts C_ji C_ik / C_ii;
    observing n_i=0 adds C_ji C_ik / (1-C_ii). This rank-one update
    avoids a QR decomposition after each occupation measurement.
    """
    K = tc.backend
    n = randoms.shape[0]

    def measure(carry, i):
        current, occupation = carry
        probability = K.clip(K.real(current[i, i]), 0.0, 1.0)
        value = randoms[i] >= 1 - probability
        selected = K.where(value, probability, 1 - probability)
        column = current[:, i]
        current = current + (K.where(value, -1.0, 1.0) / selected) * (
            column[:, None] * K.conj(column[None, :])
        )
        mask = K.cast(K.arange(n) != i, "float64")
        current = current * mask[:, None] * mask[None, :]
        current = current + K.cast(value, "float64") * (
            (1 - mask)[:, None] * (1 - mask)[None, :]
        )
        occupation = K.scatter(
            occupation, K.reshape(i, (1, 1)), K.reshape(K.cast(value, "int32"), (1,))
        )
        return current, occupation

    _, occupation = K.scan(
        measure, K.arange(n), (correlation, K.zeros((n,), dtype="int32"))
    )
    return occupation


def determinant(orbitals, occupation, particles):
    K = tc.backend
    indices = K.sort(K.argsort(-K.cast(occupation, "int32"))[:particles])
    return K.det(K.gather1d(orbitals, indices))


def importance_statistics(ratios, configurations):
    """Raw self-normalized PEPS density and delete-block jackknife uncertainty."""
    ratios = ratios / np.max(np.abs(ratios))
    weights = np.abs(ratios) ** 2
    weighted = weights[:, None] * configurations
    total = np.concatenate(([ratios.sum(), weights.sum()], weighted.sum(axis=0)))

    def moments(sums, count):
        fidelity = np.abs(sums[0]) ** 2 / (count * sums[1].real)
        return np.concatenate(([1 - fidelity], sums[2:].real / sums[1].real))

    estimate = moments(total, len(ratios))
    blocks = np.array_split(np.arange(len(ratios)), 32)
    deleted = []
    for indices in blocks:
        removed = np.concatenate(
            (
                [ratios[indices].sum(), weights[indices].sum()],
                weighted[indices].sum(axis=0),
            )
        )
        deleted.append(moments(total - removed, len(ratios) - len(indices)))
    deleted = np.asarray(deleted)
    stderr = np.sqrt(31 / 32 * np.sum((deleted - deleted.mean(axis=0)) ** 2, axis=0))
    return estimate, stderr, weights


def check_overlap(
    problem, theta, physical_time, samples=8192, seed=8201, batch_size=256
):
    """FGS only proposes validation samples; every plotted PEPS weight uses its own amplitude."""
    K = tc.backend
    alpha, orbitals = gaussian_reference(problem, physical_time)
    correlation = orbitals @ K.adjoint(orbitals)
    reference_density = np.real(np.diag(np.asarray(correlation)))
    fgs = tc.FGSSimulator(problem.nsites, alpha=alpha)
    np.testing.assert_allclose(
        np.asarray(fgs.get_cmatrix())[problem.nsites :, problem.nsites :],
        np.asarray(K.transpose(correlation)),
        atol=1e-12,
    )
    proposal = K.jit(K.vmap(lambda u: born_draw(correlation, u)))
    reference = K.jit(K.vmap(lambda s: determinant(orbitals, s, problem.particles)))
    amplitude = K.jit(problem.peps.batch_amplitude)
    rng = np.random.default_rng(seed)
    ratios, configurations = [], []
    started = time.perf_counter()
    for _ in range((samples + batch_size - 1) // batch_size):
        states = proposal(K.convert_to_tensor(rng.random((batch_size, problem.nsites))))
        np.testing.assert_array_equal(np.asarray(states).sum(axis=1), problem.particles)
        values = np.asarray(amplitude(theta, states)) / np.asarray(reference(states))
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite importance ratios")
        ratios.append(values)
        configurations.append(np.asarray(states))
    ratios = np.concatenate(ratios)[:samples]
    configurations = np.concatenate(configurations)[:samples]
    estimate, stderr, weights = importance_statistics(ratios, configurations)
    density = configurations.mean(axis=0)
    error = configurations.std(axis=0, ddof=1) / np.sqrt(samples)
    np.testing.assert_array_less(abs(density - reference_density), 6 * error + 0.002)
    difference = estimate[1:] - reference_density
    report = {
        "time": physical_time,
        "samples": samples,
        "seed": seed,
        "batch_size": batch_size,
        "proposal": "Exact Gaussian conditional rank-one updates, validated against FGSSimulator",
        "estimated_infidelity": float(estimate[0]),
        "jackknife_stderr": float(stderr[0]),
        "density_rmse": float(np.sqrt(np.mean(difference**2))),
        "density_max_error": float(np.max(abs(difference))),
        "density_stderr_rms": float(np.sqrt(np.mean(stderr[1:] ** 2))),
        "density_stderr_max": float(np.max(stderr[1:])),
        "weight_effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
        "max_weight_fraction": float(weights.max() / weights.sum()),
        "seconds_including_compilation": time.perf_counter() - started,
        "estimator": "Self-normalized importance density; no reference-density correction or smoothing.",
    }
    arrays = {
        "configurations": configurations,
        "ratios": ratios,
        "density": estimate[1:],
        "density_stderr": stderr[1:],
        "exact_density": reference_density,
    }
    return report, arrays


def check_small():
    """Check Gaussian phases and the overlap estimator against a native sector sum."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    theta = peps.random_parameters(K.get_random_state(73))
    basis, h = sector_reference(problem, -1.0)
    alpha, orbitals = gaussian_reference(problem, 0.0)
    uniforms = K.convert_to_tensor(np.random.default_rng(13).random((4096, 6)))
    for physical_time in (0.0, 0.7):
        alpha, occupied = gaussian_reference(problem, physical_time)
        correlation = occupied @ K.adjoint(occupied)
        actual = K.jit(K.vmap(lambda u: born_draw(correlation, u)))(uniforms)
        expected = K.jit(K.vmap(lambda u: born_draw_fgs(alpha, u)))(uniforms)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    reference = np.asarray(
        K.vmap(lambda s: determinant(orbitals, s, 4))(K.convert_to_tensor(basis))
    )
    energy = float(K.sum(K.eigh(problem.one_body(-1.0))[0][:4]))
    np.testing.assert_allclose(np.vdot(reference, reference), 1.0, atol=1e-12)
    np.testing.assert_allclose(h @ reference, energy * reference, atol=1e-11)
    psi = np.asarray(K.jit(peps.batch_amplitude)(theta, K.convert_to_tensor(basis)))
    exact = 1 - abs(np.vdot(reference, psi)) ** 2 / np.vdot(psi, psi).real
    report, arrays = check_overlap(problem, theta, 0.0, samples=16384)
    np.testing.assert_allclose(
        report["estimated_infidelity"], exact, atol=6 * report["jackknife_stderr"]
    )
    exact_density = np.abs(psi) ** 2 @ basis / np.vdot(psi, psi).real
    np.testing.assert_array_less(
        np.abs(arrays["density"] - exact_density), 6 * arrays["density_stderr"] + 0.002
    )
    return {
        **report,
        "exact_infidelity": float(exact),
        "fgs_matching_configurations": 8192,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--contraction-path", type=Path)
    parser.add_argument("--exact-optimizer", default="omeco-8-48")
    parser.add_argument("--samples", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=8201)
    parser.add_argument("--small-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 64 or args.batch_size < 1:
        parser.error("Use at least 64 samples and a positive batch size")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    if args.small_check:
        print(json.dumps(check_small(), indent=2))
        return
    if args.checkpoint is None or args.output is None:
        parser.error("Provide --checkpoint and --output, or use --small-check")
    saved = np.load(args.checkpoint)
    config = json.loads(str(saved["config"]))
    rows, columns, bond = (config[name] for name in ("rows", "columns", "bond_dim"))
    physical_time = (
        float(saved["step"]) * config["dt"] if config["stage"] == "real_time" else 0.0
    )
    peps = FermionPEPS(
        rows,
        columns,
        bond,
        exact_optimizer=args.exact_optimizer,
        path_file=args.contraction_path,
    )
    report, arrays = check_overlap(
        Hofstadter(peps, 2 * peps.nsites // 3),
        tc.backend.convert_to_tensor(saved["theta"]),
        physical_time,
        args.samples,
        args.seed,
        args.batch_size,
    )
    report["checkpoint_sha256"] = hashlib.sha256(
        args.checkpoint.read_bytes()
    ).hexdigest()
    report["contraction"] = "exact single-layer PEPS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
