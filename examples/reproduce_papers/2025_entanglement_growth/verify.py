"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Independent small-system checks for the example, using full-space references.
"""

import argparse
from itertools import combinations
from math import comb

import numpy as np
from accelerated import JaxKernels
from physics import (
    Entropy,
    Evolution,
    Floquet,
    SectorGates,
    gate,
    hamiltonian,
    sector_indices,
)
from scipy.linalg import expm

import tensorcircuit as tc


def verify(length):
    """Check Hamiltonians, evolution, all gate types, entropy and SWAP invariance."""
    rng = np.random.default_rng(91)
    indices = sector_indices(length)
    psi = rng.normal(size=len(indices)) + 1j * rng.normal(size=len(indices))
    psi /= np.linalg.norm(psi)
    full = np.zeros(2**length, dtype=complex)
    full[indices] = psi
    spin = [
        np.array([[0, 1], [1, 0]]) / 2,
        np.array([[0, -1j], [1j, 0]]) / 2,
        np.diag([1, -1]) / 2,
    ]
    fields = rng.uniform(-5, 5, length)

    def product(operators):
        result = np.ones((1, 1))
        for site in range(length):
            result = np.kron(result, operators.get(site, np.eye(2)))
        return result

    for jz in (0, 0.5, 1):
        reference = np.zeros((2**length, 2**length), dtype=complex)
        for site in range(length - 1):
            for operator, weight in zip(spin, (1, 1, jz)):
                reference += weight * product({site: operator, site + 1: operator})
        for site, field in enumerate(fields):
            reference += field * product({site: spin[2]})
        np.testing.assert_allclose(
            hamiltonian(length, fields, jz),
            reference[np.ix_(indices, indices)],
            atol=1e-13,
        )
    matrix = hamiltonian(length, fields)
    evolution = Evolution(matrix)
    for time in (0, 0.25, 4.5, 32):
        np.testing.assert_allclose(
            next(evolution.at(psi, [time])), expm(-1j * time * matrix) @ psi, atol=2e-12
        )
    np.testing.assert_allclose(np.linalg.norm(next(evolution.at(psi, [1e15]))), 1)

    mapper = SectorGates(length)
    for alpha, beta in (
        (0, np.pi),
        (np.pi / 2, 0),
        (np.pi, 0),
        (np.pi, np.pi / 2),
        (np.pi, np.pi),
        (np.pi / 2, np.pi),
    ):
        unitary = gate(alpha, beta)
        expected = expm(
            -1j
            * (
                alpha * (np.kron(spin[0], spin[0]) + np.kron(spin[1], spin[1]))
                + beta * np.kron(spin[2], spin[2])
            )
        )
        np.testing.assert_allclose(unitary, expected, atol=1e-13)
        for bond in range(length - 1):
            circuit = tc.Circuit(length, inputs=full)
            circuit.any(bond, bond + 1, unitary=unitary)
            np.testing.assert_allclose(
                mapper.apply(psi, unitary, bond), circuit.state()[indices], atol=1e-13
            )
    # A non-symmetric charge-conserving gate catches accidental transposition.
    circuit = tc.Circuit(2)
    circuit.rz(0, theta=0.37)
    circuit.any(0, 1, unitary=gate(np.pi / 2, np.pi / 3))
    unitary = circuit.matrix()
    for bond in range(length - 1):
        reference = tc.Circuit(length, inputs=full)
        reference.any(bond, bond + 1, unitary=unitary)
        expected = reference.state()[indices]
        np.testing.assert_allclose(
            mapper.apply(psi, unitary, bond), expected, atol=1e-13
        )
        states = np.column_stack((psi, 1j * psi))
        np.testing.assert_allclose(
            mapper.apply(states, unitary, bond),
            np.column_stack((expected, 1j * expected)),
            atol=1e-13,
        )
    entropy = Entropy(length)
    for cut in combinations(range(length), length // 2):
        rest = tuple(site for site in range(length) if site not in cut)
        matrix = (
            full.reshape([2] * length)
            .transpose(cut + rest)
            .reshape(2 ** (length // 2), -1)
        )
        probabilities = np.linalg.svd(matrix, compute_uv=False) ** 2
        probabilities = probabilities[probabilities > 0]
        expected = -np.sum(probabilities * np.log2(probabilities))
        np.testing.assert_allclose(entropy.value(psi, cut), expected, atol=1e-12)
        # TC's generic helper uses natural logs and a small density regularizer.
        np.testing.assert_allclose(
            entropy.value(psi, cut),
            tc.quantum.entanglement_entropy(full, subsystem_to_keep=cut) / np.log(2),
            atol=1e-9,
            rtol=0,
        )
        np.testing.assert_allclose(
            entropy.value(psi, cut), entropy.value(psi, rest), atol=1e-12
        )
    cuts = entropy.cuts()
    assert len(cuts) == comb(length, length // 2) // 2
    before = entropy.average(psi, cuts)
    state = psi.copy()
    for bond in rng.permutation(length - 1):
        state = mapper.apply(state, gate(np.pi, np.pi), bond)
        np.testing.assert_allclose(entropy.average(state, cuts), before, atol=1e-12)
    # One Bell pair crossing the middle cut fixes the logarithm convention.
    bell = np.zeros(len(sector_indices(4)), dtype=complex)
    bell[np.isin(sector_indices(4), [0b0101, 0b0011])] = 1 / np.sqrt(2)
    np.testing.assert_allclose(Entropy(4).value(bell), 1, atol=1e-12)
    floquet = Floquet(length, fields)
    expected = expm(-1j * hamiltonian(length, fields, jz=1, jxy=0)) @ expm(
        -0.4j * hamiltonian(length, np.zeros(length), jz=0)
    )
    for period in (0, 1, 3, 10):
        np.testing.assert_allclose(
            next(floquet.at(psi, [period])),
            np.linalg.matrix_power(expected, period) @ psi,
            atol=2e-12,
        )
    print(
        f"L={length}: passed XXZ, evolution, TC gates, entropy, SWAP, and Floquet checks."
    )


def verify_acceleration(length):
    """Check compiled batches and late-window scans against the NumPy path."""
    rng = np.random.default_rng(671)
    entropy = Entropy(length)
    states = rng.normal(size=(len(entropy.idx), 3)) + 1j * rng.normal(
        size=(len(entropy.idx), 3)
    )
    states[:, 0] = 0
    states[0, 0] = 1
    states /= np.linalg.norm(states, axis=0)
    kernels = JaxKernels(length)
    np.testing.assert_allclose(
        kernels.hcee(states),
        [entropy.value(s) for s in states.T],
        atol=2e-12,
        rtol=1e-11,
    )
    cuts = entropy.cuts()
    np.testing.assert_allclose(
        kernels.baee(states, cuts),
        entropy.average_many(states, cuts),
        atol=2e-12,
        rtol=1e-11,
    )
    # 107 layers cover the warmup/late-window boundary in the two scans.
    bonds = rng.integers(0, length - 1, (2, 107))
    mapper = SectorGates(length)
    unitaries = [
        gate(a, b)
        for a, b in (
            (np.pi / 2, np.pi),
            (0, np.pi),
            (np.pi / 2, 0),
            (np.pi, 0),
            (np.pi, np.pi / 2),
            (np.pi, np.pi),
        )
    ]
    circuit = tc.Circuit(2)
    circuit.rz(0, theta=0.37)
    circuit.any(0, 1, unitary=gate(np.pi / 2, np.pi / 3))
    unitaries.append(circuit.matrix())
    K = kernels.K
    for unitary in unitaries:
        finals, means = kernels.circuit(
            K.convert_to_tensor(states.T),
            K.convert_to_tensor(unitary),
            K.convert_to_tensor(bonds),
        )
        finals, means = K.numpy(finals), K.numpy(means)
        for realization, sequence in enumerate(bonds):
            state, total = states.copy(), np.zeros(states.shape[1])
            for depth, bond in enumerate(sequence):
                state = mapper.apply(state, unitary, bond)
                if depth >= len(sequence) - 100:
                    total += [entropy.value(s) for s in state.T]
            np.testing.assert_allclose(
                finals[realization], state.T, atol=2e-12, rtol=1e-11
            )
            np.testing.assert_allclose(
                means[realization], total / 100, atol=2e-12, rtol=1e-11
            )
    print(f"L={length}: passed JAX entropy batches, all cuts, and circuit scans.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-only", action="store_true")
    args = parser.parse_args()
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        for length in (4, 6):
            verify(length)
        if not args.numpy_only:
            for length in (4, 6, 8):
                verify_acceleration(length)
