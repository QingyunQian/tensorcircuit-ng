"""
Independent small-system checks for fermionic PEPS, Monte Carlo and tVMC.

Full Hilbert-space calculations occur only in this validation script.
TenCirPauli supplies CAR algebra and sector Hamiltonians; TensorCircuit
supplies the statevector action and the differentiable PEPS kernels.
"""

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import tencirpauli as tcp
import tensorcircuit as tc

if __package__:
    from ...peps_boundary_mps import (
        equilibrated_qr,
        precompute_L_envs,
        precompute_R_envs,
        sweep_left,
        sweep_right,
    )
    from .peps import FermionPEPS, Hofstadter
    from .run_peps_6x6 import enlarge_bond
    from .tvmc import MonteCarlo, minsr_solve, sr_solve, sr_cg_solve
else:
    from peps_boundary_mps import (
        equilibrated_qr,
        precompute_L_envs,
        precompute_R_envs,
        sweep_left,
        sweep_right,
    )
    from peps import FermionPEPS, Hofstadter
    from run_peps_6x6 import enlarge_bond
    from tvmc import MonteCarlo, minsr_solve, sr_solve, sr_cg_solve


def sector_reference(problem, potential=0.0):
    """Use TenCirPauli's native restricted operator without rebuilding CAR signs."""
    space = tcp.OperatorSpace(fermions=problem.nsites)
    number = tcp.AdditiveCharge(
        space, name="N", fermions={i: 1 for i in range(problem.nsites)}
    )
    sector = tcp.ChargeSector(((number, problem.particles),))
    operator = problem.operator.add(problem.pin_operator.scale(potential))
    return sector.basis_states(), operator.restrict_charge(sector).dense()


def swap_network_reference(peps, theta, occupation):
    """Explicit virtual-bond sum with individual physical/virtual swap factors."""
    tensors = [np.asarray(tc.backend.numpy(t)) for t in peps.tensors(theta)]
    amplitude = 0j
    for bonds in itertools.product(range(peps.bond_dim), repeat=len(peps.bonds)):
        indices = [[0, 0, 0, 0] for _ in tensors]
        for value, (i, j, ai, aj) in zip(bonds, peps.bonds):
            indices[i][ai - 1] = value
            indices[j][aj - 1] = value
        value = 1 + 0j
        for i, tensor in enumerate(tensors):
            value *= tensor[(occupation[i],) + tuple(indices[i])]
        # Draw each physical leg toward the upper-left boundary of the grid.
        for i, n in enumerate(occupation):
            y, x = divmod(i, peps.columns)
            for k in range(x):
                crossed = y * peps.columns + k
                value *= (-1) ** (int(n) * indices[crossed][3])
        amplitude += value
    return amplitude


def check_boundary_qr():
    """Check batched QR across extreme column scales and exactly null bonds."""
    K = tc.backend
    rng = np.random.default_rng(7301)
    matrix = rng.normal(size=(64, 12, 6)) + 1j * rng.normal(size=(64, 12, 6))
    scales = np.array([1.0, 1e-16, 1e-32, 1e-100, 1e-200, 0.0])
    values = matrix * scales
    q, r = map(np.asarray, K.jit(K.vmap(equilibrated_qr))(K.convert_to_tensor(values)))
    denominator = np.where(scales > 0, scales, 1.0)
    reconstructed = q @ (r / denominator)
    np.testing.assert_allclose(reconstructed, values / denominator, atol=1e-12)
    identity = np.broadcast_to(np.eye(6), (64, 6, 6))
    np.testing.assert_allclose(q.conj().transpose(0, 2, 1) @ q, identity, atol=1e-12)
    direction = rng.normal(size=(12, 6)) + 1j * rng.normal(size=(12, 6))

    def loss(value):
        left, right = equilibrated_qr(value)
        return K.sum(K.real(left) ** 2) + 0.1 * K.sum(K.abs(right) ** 2)

    value = K.convert_to_tensor(matrix[0])
    derivative = np.asarray(K.jit(K.grad(loss))(value))
    step = 1e-5
    objective = K.jit(loss)
    finite = float(
        (objective(value + step * direction) - objective(value - step * direction))
        / (2 * step)
    )
    automatic = float(np.real(np.sum(derivative * direction)))
    np.testing.assert_allclose(automatic, finite, rtol=1e-7, atol=1e-8)
    return {
        "directional_derivative_error": abs(automatic - finite),
        "max_scaled_reconstruction_error": float(
            np.max(np.abs(reconstructed - values / denominator))
        ),
    }


def check_sweep_environments():
    """Retained carries must equal contractions of the updated MPS, including AD."""
    K = tc.backend
    rng = np.random.default_rng(271)
    previous = 0.2 * (
        rng.normal(size=(3, 2, 2, 2)) + 1j * rng.normal(size=(3, 2, 2, 2))
    )
    grid = 0.2 * (
        rng.normal(size=(3, 2, 2, 2, 2)) + 1j * rng.normal(size=(3, 2, 2, 2, 2))
    )
    previous[0, 1:] = previous[-1, :, :, 1:] = 0
    grid[0, :, :, 1:] = grid[-1, :, :, :, 1:] = 0
    previous, grid = K.convert_to_tensor(previous), K.convert_to_tensor(grid)
    boundary = np.zeros((2, 2, 2), dtype=complex)
    boundary[0, 0, 0] = 1
    boundary = K.convert_to_tensor(boundary)

    def environments(value, reuse):
        right = precompute_R_envs(previous, previous, value, boundary)
        updated, left = sweep_right(
            previous, previous, value, boundary, right, return_envs=True
        )
        if not reuse:
            left = precompute_L_envs(updated, previous, value, boundary)
        updated, right = sweep_left(
            updated, previous, value, left, boundary, return_envs=True
        )
        if not reuse:
            right = precompute_R_envs(updated, previous, value, boundary)
        return K.stack([left, right])

    results, gradients = [], []
    for reuse in (False, True):
        results.append(np.asarray(K.jit(lambda g: environments(g, reuse))(grid)))
        loss = lambda g: K.sum(K.abs(environments(g, reuse)) ** 2)
        gradients.append(np.asarray(K.jit(K.grad(loss))(grid)))
    np.testing.assert_allclose(*results, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(*gradients, atol=1e-11, rtol=1e-10)
    return {
        "max_environment_error": float(np.max(abs(results[0] - results[1]))),
        "max_gradient_error": float(np.max(abs(gradients[0] - gradients[1]))),
    }


def check_kernels(rows=2, columns=3, bond_dim=2):
    K = tc.backend
    peps = FermionPEPS(rows, columns, bond_dim)
    problem = Hofstadter(peps, 2 * peps.nsites // 3)
    theta = peps.random_parameters(K.get_random_state(71))
    basis, hamiltonian = sector_reference(problem)
    basis = K.cast(K.convert_to_tensor(basis), "int64")
    amplitude, scores = K.jit(peps.batch_scores)(theta, basis)
    psi, jac = [
        np.asarray(K.numpy(x)) for x in (amplitude, amplitude[:, None] * scores)
    ]
    local = np.asarray(K.numpy(K.jit(problem.local_batch)(theta, basis)))
    np.testing.assert_allclose(local * psi, hamiltonian @ psi, rtol=1e-10, atol=1e-12)
    if rows * columns <= 6 and bond_dim == 2:
        explicit = np.array(
            [swap_network_reference(peps, theta, np.asarray(s)) for s in basis]
        )
        np.testing.assert_allclose(psi, explicit, rtol=1e-11, atol=1e-13)
    # Direct circuit-state MVP checks both the native restricted basis order and
    # the mapped Pauli convention used by local-energy evaluation.
    full = np.zeros(2**peps.nsites, dtype=np.complex128)
    powers = 2 ** np.arange(peps.nsites - 1, -1, -1)
    labels = np.asarray(basis) @ powers
    full[labels] = psi
    circuit = tc.Circuit(peps.nsites, inputs=K.convert_to_tensor(full))
    mvp = tcp.backend_mvp(problem.pauli.backend_mvp_plan(), backend=K)
    acted = np.asarray(K.numpy(mvp(circuit.state())))
    np.testing.assert_allclose(acted[labels], hamiltonian @ psi, rtol=1e-11, atol=1e-13)
    direction = peps.random_parameters(K.get_random_state(97))
    eps = 1e-6
    amplitudes = K.jit(peps.batch_amplitude)
    finite = np.asarray(
        K.numpy(
            (
                amplitudes(theta + eps * direction, basis)
                - amplitudes(theta - eps * direction, basis)
            )
            / (2 * eps)
        )
    )
    np.testing.assert_allclose(
        finite, jac @ np.asarray(direction), rtol=1e-7, atol=1e-10
    )
    vectors = np.asarray(K.numpy(peps.gauge_vectors(theta)))
    centered = np.asarray(scores) - np.mean(np.asarray(scores), axis=0)
    gauge_error = np.linalg.norm(centered @ vectors) / np.linalg.norm(centered)
    np.testing.assert_allclose(centered @ vectors, 0.0, atol=2e-10)
    projector = K.jit(peps.gauge_projector)(theta)
    np.testing.assert_allclose(projector @ vectors, 0.0, atol=2e-10)
    weights = K.abs(amplitude) ** 2 / K.sum(K.abs(amplitude) ** 2)
    velocity, _, residual = sr_solve(
        scores, K.convert_to_tensor(local), weights, projector
    )
    thin_velocity, _, _ = sr_solve(
        scores, K.convert_to_tensor(local), weights, gauge_basis=peps.gauge_basis(theta)
    )
    np.testing.assert_allclose(
        jac @ np.asarray(thin_velocity),
        jac @ np.asarray(velocity),
        rtol=2e-7,
        atol=2e-9,
    )
    cg_velocity, _, _, linear_residual, _ = K.jit(sr_cg_solve)(
        scores, K.convert_to_tensor(local), weights
    )
    np.testing.assert_allclose(
        jac @ np.asarray(cg_velocity), jac @ np.asarray(velocity), rtol=3e-6, atol=3e-8
    )
    np.testing.assert_allclose(
        vectors.conj().T @ np.asarray(cg_velocity), 0.0, atol=2e-7
    )
    np.testing.assert_allclose(linear_residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(vectors.conj().T @ np.asarray(velocity), 0.0, atol=2e-8)
    return {
        "shape": [rows, columns],
        "D": bond_dim,
        "parameters": peps.nparams,
        "local_energy_max_error": float(
            np.max(np.abs(local * psi - hamiltonian @ psi))
        ),
        "gauge_null_error": float(gauge_error),
        "sr_residual": float(residual),
    }


def check_boundary():
    """The reused approximate contractor and hole scores approach exact contraction."""
    K = tc.backend
    exact = FermionPEPS(2, 3, 2)
    approx = FermionPEPS(2, 3, 2, boundary_dim=4)
    theta = exact.random_parameters(K.get_random_state(72))
    occupation = K.convert_to_tensor([1, 1, 0, 1, 0, 1])
    a, da = K.jit(exact.scores)(theta, occupation)
    b, db = K.jit(approx.scores)(theta, occupation)
    np.testing.assert_allclose(b, a, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(db, da, rtol=2e-6, atol=2e-6)
    return {
        "amplitude_relative_error": float(K.abs((a - b) / a)),
        "score_relative_error": float(K.norm(da - db) / K.norm(da)),
    }


def check_boundary_convergence():
    """Increase the reused boundary-MPS cap until a nontrivial cut is exact."""
    K = tc.backend
    exact = FermionPEPS(3, 4, 2)
    theta = exact.random_parameters(K.get_random_state(79))
    occupation = K.convert_to_tensor([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    amplitude, score = K.jit(exact.scores)(theta, occupation)
    errors = []
    for chi in (2, 4):
        approx = FermionPEPS(3, 4, 2, boundary_dim=chi)
        value, derivative = K.jit(approx.scores)(theta, occupation)
        errors.append(
            [
                float(K.abs((value - amplitude) / amplitude)),
                float(K.norm(derivative - score) / K.norm(score)),
            ]
        )
    np.testing.assert_allclose(errors[-1], 0.0, atol=1e-5)
    return {"boundary_dims": [2, 4], "amplitude_and_score_relative_errors": errors}


def check_cg():
    """Weighted complex ridge equations, rank deficiency, and zero-force behavior."""
    K = tc.backend
    rng = np.random.default_rng(102)
    errors = []
    for samples, parameters in ((9, 24), (41, 12)):
        scores = rng.normal(size=(samples, parameters)) + 1j * rng.normal(
            size=(samples, parameters)
        )
        scores[:, -1] = scores[:, 0]
        energies = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        weights = rng.uniform(size=samples)
        weights[0] = 0.0
        weights /= weights.sum()
        centered = scores - weights @ scores
        force = centered.conj().T @ (weights * (energies - weights @ energies))
        metric = centered.conj().T @ (weights[:, None] * centered)
        reference = np.linalg.solve(metric + 1e-3 * np.eye(parameters), force)
        values = tuple(K.convert_to_tensor(a) for a in (scores, energies, weights))
        solve = K.jit(lambda o, e, w: sr_cg_solve(o, e, w, 1e-3, 1e-10))
        velocity, _, _, residual, iterations = solve(*values)
        np.testing.assert_allclose(velocity, reference, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

        def loss(scale):
            direction = sr_cg_solve(
                values[0] * scale, values[1], values[2], 1e-3, 1e-10
            )[0]
            return K.real(K.sum(K.conj(direction) * direction))

        derivative = K.jit(K.grad(loss))(K.cast(1.0, "float64"))
        epsilon = 1e-5
        norms = []
        for scale in (1 + epsilon, 1 - epsilon):
            direction = np.linalg.solve(
                scale**2 * metric + 1e-3 * np.eye(parameters), scale * force
            )
            norms.append(np.vdot(direction, direction).real)
        np.testing.assert_allclose(
            derivative, (norms[0] - norms[1]) / (2 * epsilon), rtol=2e-6, atol=1e-8
        )
        failed = sr_cg_solve(*values, regulator=1e-3, tolerance=1e-10, maxiter=1)
        assert float(failed[3]) > 1e-5
        zero = solve(values[0], K.zeros((samples,), dtype="complex128"), values[2])
        np.testing.assert_allclose(zero[0], 0.0, atol=0)
        np.testing.assert_allclose(zero[3], 0.0, atol=0)
        errors.append(
            {
                "N": samples,
                "P": parameters,
                "iterations": int(iterations),
                "relative_residual": float(residual),
            }
        )
    return errors


def check_cg_estimate():
    """CG dispatch and RK stages agree with gauge-projected SR on exact scores."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2, exact_optimizer="greedy")
    problem = Hofstadter(peps, 4)
    cg = MonteCarlo(problem, chains=32, draws=1, regulator=1e-3)
    dense = MonteCarlo(problem, chains=32, draws=1, solver="dense", regulator=1e-3)
    theta = peps.random_parameters(K.get_random_state(104))
    states = cg.initial_chains(K.get_random_state(105))
    key = K.get_random_state(106)
    estimate = K.jit(cg.estimate)(theta, states, key)
    reference = K.jit(dense.estimate)(theta, states, key)
    for value, expected in zip(estimate, reference):
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-7)
    step = K.jit(lambda p, s, k: cg.rk4_step(p, s, k, 0.001))(theta, states, key)
    expected = K.jit(lambda p, s, k: dense.rk4_step(p, s, k, 0.001))(theta, states, key)
    for value, reference in zip(step, expected):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)
    failed = MonteCarlo(problem, chains=32, draws=1, regulator=1e-3, cg_maxiter=1)
    rejected = K.jit(failed.estimate)(theta, states, key)
    assert np.isnan(np.asarray(rejected[0])).all()
    return {"estimate_and_rk4_agree": True, "nonconverged_direction_rejected": True}


def check_exact_path():
    """Saved paths preserve amplitudes and complex gradients without a path finder."""
    K = tc.backend
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "path.json"
        planned = FermionPEPS(2, 3, 2, exact_optimizer="greedy", path_file=path)
        cached = FermionPEPS(
            2, 3, 2, exact_optimizer="unused-on-cache-hit", path_file=path
        )
        theta = planned.random_parameters(K.get_random_state(903))
        occupation = K.convert_to_tensor([1, 1, 0, 1, 1, 0])
        first = K.jit(planned.scores)(theta, occupation)
        second = K.jit(cached.scores)(theta, occupation)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            first[0],
            swap_network_reference(planned, theta, occupation),
            rtol=1e-12,
            atol=1e-12,
        )
        try:
            FermionPEPS(3, 3, 2, path_file=path)
        except ValueError as error:
            assert "topology" in str(error)
        else:
            raise AssertionError("A mismatched topology was accepted.")
    return {
        "cache_amplitude_and_score_agree": True,
        "mismatched_topology_rejected": True,
    }


def check_minsr():
    """The published sample-space solve satisfies the sampled SR equation."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    theta = peps.random_parameters(K.get_random_state(83))
    basis, _ = sector_reference(problem)
    basis = K.cast(K.convert_to_tensor(basis[:6]), "int64")
    amplitudes, scores = K.jit(peps.batch_scores)(theta, basis)
    energies = K.jit(problem.local_batch)(theta, basis)
    weights = K.abs(amplitudes) ** 2 / K.sum(K.abs(amplitudes) ** 2)
    velocity, _, residual = minsr_solve(scores, energies, weights)
    np.testing.assert_allclose(scores @ velocity, energies, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    return {
        "sampled_rows": 6,
        "parameters": peps.nparams,
        "sr_residual": float(residual),
    }


def check_hopping_boundaries():
    """Check every hopping orientation, including vertical swap strings."""
    K = tc.backend
    report = []
    for rows, columns, dimension, chi in ((2, 3, 4, 16), (3, 3, 2, 4), (3, 4, 2, 4)):
        exact = FermionPEPS(rows, columns, dimension)
        approx = FermionPEPS(rows, columns, dimension, chi)
        problem = Hofstadter(exact, 2 * rows * columns // 3)
        bounded = Hofstadter(approx, problem.particles)
        basis, _ = sector_reference(problem)
        states = K.cast(
            K.convert_to_tensor(np.asarray(basis)[:: max(1, len(basis) // 7)]), "int64"
        )
        theta = exact.random_parameters(K.get_random_state(93))

        def reference(params, state):
            ratios = exact.batch_amplitude(params, state[problem.permutations])
            ratios = ratios / exact.amplitude(params, state)
            return K.where(state[problem.left] != state[problem.right], ratios, 0.0)

        expected = K.jit(K.vmap(reference, vectorized_argnums=1))(theta, states)
        actual = K.jit(K.vmap(approx.boundary_hopping_ratios, vectorized_argnums=1))(
            theta, states
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(
            K.jit(bounded.local_batch)(theta, states, -1.0),
            K.jit(problem.local_batch)(theta, states, -1.0),
            rtol=2e-7,
            atol=2e-7,
        )
        report.append(
            {
                "shape": [rows, columns],
                "D": dimension,
                "chi": chi,
                "max_ratio_error": float(K.max(K.abs(actual - expected))),
            }
        )
    return report


def check_batched_moments():
    """Contraction chunking, including the padded final chunk, preserves moments."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    sampler = MonteCarlo(problem, contraction_batch_size=4)
    basis, _ = sector_reference(problem)
    states = K.cast(K.convert_to_tensor(basis), "int64")
    theta = peps.random_parameters(K.get_random_state(83))
    scores, energies = K.jit(sampler.sampled_moments)(theta, states, -1.0)
    np.testing.assert_allclose(
        scores, K.jit(peps.batch_scores)(theta, states)[1], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        energies,
        K.jit(problem.local_batch)(theta, states, -1.0),
        rtol=1e-10,
        atol=1e-10,
    )
    amplitudes = K.jit(peps.batch_amplitude)(theta, states)
    probabilities = K.abs(amplitudes) ** 2
    for nonzero in (len(basis), 5):
        weights = K.where(K.arange(len(basis)) < nonzero, probabilities, 0.0)
        weights = weights / K.sum(weights)
        metric, force, energy, values = K.jit(sampler.sr_statistics)(
            theta, states, weights, -1.0
        )
        centered = scores - K.sum(weights[:, None] * scores, axis=0)
        expected_energy = K.sum(weights * energies)
        np.testing.assert_allclose(
            metric,
            K.adjoint(centered) @ (weights[:, None] * centered),
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            force,
            K.adjoint(centered) @ (weights * (energies - expected_energy)),
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(values, energies, rtol=1e-11, atol=1e-11)
    return {"configurations": len(basis), "contraction_batch_size": 4}


def check_sampling(boundary_dim=None):
    """Born frequencies and moments are checked against independent exact sums."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2, boundary_dim)
    problem = Hofstadter(peps, 4)
    theta = peps.random_parameters(K.get_random_state(41))
    sampler = MonteCarlo(problem, chains=256, draws=128, sweeps=4)
    chains = sampler.initial_chains(K.get_random_state(43))
    chains, key, _ = K.jit(lambda t, s, k: sampler.advance(t, s, k, 512))(
        theta, chains, K.get_random_state(47)
    )
    samples, _, _, acceptance = K.jit(sampler.sample)(theta, chains, key)
    samples = np.asarray(K.numpy(samples))
    np.testing.assert_array_equal(samples.sum(axis=1), 4)
    basis, _ = sector_reference(problem)
    amplitudes = np.asarray(
        K.numpy(peps.batch_amplitude(theta, K.convert_to_tensor(basis)))
    )
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= probabilities.sum()
    density = probabilities @ basis
    block_means = samples.reshape(128, 256, 6).mean(axis=0)
    stderr = block_means.std(axis=0, ddof=1) / np.sqrt(256)
    np.testing.assert_allclose(samples.mean(axis=0), density, atol=0.025)
    np.testing.assert_array_less(
        np.abs(samples.mean(axis=0) - density), 6 * stderr + 0.002
    )
    return {
        "samples": int(len(samples)),
        "acceptance": float(acceptance),
        "density_max_error": float(np.max(np.abs(samples.mean(axis=0) - density))),
    }


class ExactMoments(MonteCarlo):
    """Deterministic validation of the production RK4 integrator on a tiny lattice."""

    def __init__(self, problem):
        super().__init__(problem, chains=2, draws=1, regulator=1e-10)
        self.basis, self.hamiltonian = sector_reference(problem)
        self.basis = tc.backend.cast(tc.backend.convert_to_tensor(self.basis), "int64")

    def estimate(self, theta, chains, key, potential=0.0):
        K = tc.backend
        amplitude, scores = self.peps.batch_scores(theta, self.basis)
        energies = self.problem.local_batch(theta, self.basis, potential)
        weights = K.abs(amplitude) ** 2 / K.sum(K.abs(amplitude) ** 2)
        velocity, energy, residual = sr_solve(
            scores, energies, weights, self.peps.gauge_projector(theta), self.regulator
        )
        diagnostic = K.zeros((4 + 2 * self.problem.nsites,), dtype="float64")
        diagnostic = K.scatter(
            diagnostic,
            K.convert_to_tensor([[0], [2]]),
            K.stack([K.real(energy), residual]),
        )
        return velocity, diagnostic, chains, key


def check_preparation():
    """Check parity-preserving growth and the sign of the Euler VMC update."""
    K = tc.backend
    peps, larger = FermionPEPS(2, 3, 2), FermionPEPS(2, 3, 4)
    problem = Hofstadter(peps, 4)
    engine = ExactMoments(problem)
    theta = peps.random_parameters(K.get_random_state(141))
    grown = enlarge_bond(theta, peps, larger, noise=0.0)
    before = np.asarray(K.jit(peps.batch_amplitude)(theta, engine.basis))
    after = np.asarray(K.jit(larger.batch_amplitude)(grown, engine.basis))
    np.testing.assert_allclose(after, before, rtol=1e-11, atol=1e-14)
    chains = engine.initial_chains(K.get_random_state(143))
    key = K.get_random_state(149)
    final, _, _, _ = K.jit(
        lambda p, s, k: engine.trajectory(p, s, k, 0.001, 1, -1.0, True, True)
    )(theta, chains, key)
    psi = np.asarray(K.jit(peps.batch_amplitude)(final, engine.basis))
    _, h = sector_reference(problem, -1.0)
    initial_energy = np.vdot(before, h @ before).real / np.vdot(before, before).real
    final_energy = np.vdot(psi, h @ psi).real / np.vdot(psi, psi).real
    np.testing.assert_array_less(final_energy, initial_energy)
    return {
        "embedding_max_error": float(np.max(abs(after - before))),
        "preparation_energy_change": float(final_energy - initial_energy),
    }


def check_integrator():
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    engine = ExactMoments(problem)
    theta = peps.random_parameters(K.get_random_state(121))
    psi = np.array(K.numpy(peps.batch_amplitude(theta, engine.basis)), copy=True)
    psi /= np.linalg.norm(psi)
    exact = expm(-0.04j * engine.hamiltonian) @ psi
    errors = []
    for steps in (4, 8):
        trajectory = K.jit(
            lambda p, s, k: engine.trajectory(p, s, k, 0.04 / steps, steps)
        )
        final, _, _, _ = trajectory(
            theta,
            engine.initial_chains(K.get_random_state(131)),
            K.get_random_state(137),
        )
        state = np.array(K.numpy(peps.batch_amplitude(final, engine.basis)), copy=True)
        state /= np.linalg.norm(state)
        state *= np.exp(-1j * np.angle(np.vdot(exact, state)))
        errors.append(float(np.linalg.norm(state - exact)))
    np.testing.assert_allclose(errors, 0.0, atol=2e-5)
    np.testing.assert_array_less(errors[1], errors[0] + 1e-8)
    return {"dt": [0.01, 0.005], "state_norm_errors": errors}


def check_saved_run(directory):
    """Separate initial-state, integration, and measurement errors on small runs."""
    report = json.loads((directory / "peps_results.json").read_text())
    if report["rows"] * report["columns"] > 12:
        raise ValueError("Exact saved-run validation is limited to at most 12 sites.")
    saved = np.load(directory / "peps_results.npz")
    K = tc.backend
    peps = FermionPEPS(
        report["rows"], report["columns"], report["D"], report["boundary_dim"]
    )
    problem = Hofstadter(peps, report["particles"])
    basis, initial_h = sector_reference(problem, -1.0)
    _, hamiltonian = sector_reference(problem)
    eigenvalues, eigenvectors = np.linalg.eigh(initial_h)
    ground = eigenvectors[:, 0]
    amplitude = K.jit(peps.batch_amplitude)
    states = []
    for name in ("initial_theta", "final_theta"):
        psi = np.array(
            K.numpy(
                amplitude(K.convert_to_tensor(saved[name]), K.convert_to_tensor(basis))
            ),
            copy=True,
        )
        states.append(psi / np.linalg.norm(psi))
    initial, final = states
    evolution = expm(-1j * hamiltonian * saved["times"][-1])
    exact_from_initial = evolution @ initial
    initial_energy = float(np.vdot(initial, initial_h @ initial).real)
    final_energy = float(np.vdot(final, hamiltonian @ final).real)
    initial_variance = float(
        np.linalg.norm(initial_h @ initial - initial_energy * initial) ** 2
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    evolved = (
        np.exp(-1j * saved["times"][:, None] * energies) * (vectors.conj().T @ ground)
    ) @ vectors.T
    exact_density = np.abs(evolved) ** 2 @ basis
    np.testing.assert_allclose(saved["exact_density"], exact_density, atol=1e-11)
    np.testing.assert_allclose(
        saved["density"].sum(axis=1), problem.particles, atol=1e-10
    )
    np.testing.assert_array_less(
        np.abs(saved["density"] - exact_density),
        6 * saved["density_stderr"] + 0.002,
    )
    final_density = np.abs(final) ** 2 @ basis
    overlap = np.vdot(exact_from_initial, final)
    state_error = np.linalg.norm(
        final * np.exp(-1j * np.angle(overlap)) - exact_from_initial
    )
    np.testing.assert_array_less(
        np.abs(saved["density"][-1] - final_density),
        6 * saved["density_stderr"][-1] + 0.002,
    )
    metrics = {
        "initial_energy_error": initial_energy - float(eigenvalues[0]),
        "initial_variance": initial_variance,
        "initial_infidelity": float(1 - abs(np.vdot(ground, initial)) ** 2),
        "evolution_infidelity_from_prepared_state": float(1 - abs(overlap) ** 2),
        "evolution_state_norm_error": float(state_error),
        "final_infidelity_from_exact_ground_state": float(
            1 - abs(np.vdot(evolution @ ground, final)) ** 2
        ),
        "exact_peps_energy_drift": final_energy
        - float(np.vdot(initial, hamiltonian @ initial).real),
        "final_exact_density_error_vs_fgs": float(
            np.max(np.abs(final_density - exact_density[-1]))
        ),
        "density_sampling_rmse_vs_fgs": float(
            np.sqrt(np.mean((saved["density"] - exact_density) ** 2))
        ),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels-only", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    if args.result_dir is not None:
        report = check_saved_run(args.result_dir)
        print(json.dumps(report, indent=2), flush=True)
        np.testing.assert_allclose(report["initial_energy_error"], 0.0, atol=1e-6)
        np.testing.assert_allclose(report["evolution_state_norm_error"], 0.0, atol=1e-4)
        return
    report = {
        "kernels": [check_kernels(), check_kernels(3, 3, 2), check_kernels(3, 3, 4)]
    }
    print(json.dumps(report, indent=2), flush=True)
    if not args.kernels_only:
        report["boundary_qr"] = check_boundary_qr()
        report["sweep_environments"] = check_sweep_environments()
        report["boundary"] = check_boundary()
        report["boundary_convergence"] = check_boundary_convergence()
        report["hopping_boundaries"] = check_hopping_boundaries()
        report["batched_moments"] = check_batched_moments()
        report["sampling"] = check_sampling()
        report["sequential_sampling"] = check_sampling(4)
        report["minsr"] = check_minsr()
        report["cg"] = check_cg()
        report["cg_estimate"] = check_cg_estimate()
        report["exact_path"] = check_exact_path()
        report["integrator"] = check_integrator()
        report["preparation"] = check_preparation()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
