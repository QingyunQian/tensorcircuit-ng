"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

JAX entropy and circuit kernels; spectral preparation stays on the CPU.
"""

import numpy as np
from physics import Entropy, SectorGates

import tensorcircuit as tc


class JaxKernels:
    """Compile entropy batches and whole random-circuit trajectories once per L."""

    def __init__(self, length):
        self.K = tc.set_backend("jax", set_global=False)
        K = self.K
        self.entropy = Entropy(length)
        half = self.cut_maps(self.entropy.half)
        self.half = K.jit(K.vmap(self.entropy_at, vectorized_argnums=0))
        self.half_maps = half

        code, partner, opposite = zip(*SectorGates(length).maps)
        self.code = K.convert_to_tensor(np.array(code))
        self.partner = K.convert_to_tensor(np.array(partner))
        self.opposite = K.convert_to_tensor(np.array(opposite))

        def apply(state, unitary, bond):
            codes = self.code[bond]
            diagonal = unitary[codes, codes]
            off = unitary[codes, 3 - codes] * self.opposite[bond]
            return diagonal * state + off * state[self.partner[bond]]

        apply_many = K.vmap(apply, vectorized_argnums=0)
        entropy_many = K.vmap(self.entropy_at, vectorized_argnums=0)

        def circuit(states, unitary, bonds):
            late = min(100, bonds.shape[0])
            states = K.scan(
                lambda carry, bond: apply_many(carry, unitary, bond),
                bonds[:-late],
                states,
            )

            def measure(carry, bond):
                state, total = carry
                state = apply_many(state, unitary, bond)
                return state, total + entropy_many(state, half)

            state, total = K.scan(
                measure, bonds[-late:], (states, K.zeros((states.shape[0],), "float64"))
            )
            return state, total / late

        self.circuit = K.jit(K.vmap(circuit, vectorized_argnums=2))

        def sum_cuts(states, maps):
            return K.scan(
                lambda total, cut: total + entropy_many(states, cut),
                maps,
                K.zeros((states.shape[0],), "float64"),
            )

        self.sum_cuts = K.jit(sum_cuts)

    def cut_maps(self, cut):
        """Static amplitude indices, grouped by subsystem charge."""
        result = []
        for mask, rows, columns, shape in self.entropy.blocks(tuple(cut)):
            indices = np.empty(shape, dtype=np.int32)
            indices[rows, columns] = mask
            result.append(indices)
        return tuple(result)

    def entropy_at(self, state, maps):
        """Charge-block form of TC entanglement entropy, without regularization."""
        K = self.K
        norm = K.sum(K.abs(state) ** 2)
        entropy = K.zeros((), "float64")
        for indices in maps:
            _, singular, _, _ = K.svd(state[indices], pivot_axis=1)
            probabilities = K.abs(singular) ** 2 / norm
            safe = K.where(probabilities > 0, probabilities, K.ones_like(probabilities))
            entropy -= K.sum(probabilities * K.log(safe)) / np.log(2)
        return entropy

    def hcee(self, states):
        return self.K.numpy(
            self.half(self.K.convert_to_tensor(states.T), self.half_maps)
        )

    def baee(self, states, cuts):
        K = self.K
        states = K.convert_to_tensor(states.T)
        total = K.zeros((states.shape[0],), "float64")
        # Stream cut chunks instead of materializing all cuts and all SVDs.
        for start in range(0, len(cuts), 32):
            chunk = [self.cut_maps(cut) for cut in cuts[start : start + 32]]
            maps = tuple(K.convert_to_tensor(np.stack(x)) for x in zip(*chunk))
            total = total + self.sum_cuts(states, maps)
        return K.numpy(total / len(cuts))

    def rqc(self, states, unitary, bonds):
        K = self.K
        _, means = self.circuit(
            K.convert_to_tensor(states.T),
            K.convert_to_tensor(unitary),
            K.convert_to_tensor(bonds),
        )
        return K.numpy(K.mean(means, axis=0))
