"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Exact half-filled-sector physics for Figures 1(d,e) and 3, using the NumPy backend.
TC constructs the Pauli Hamiltonians and gates; its backend evaluates dense
linear algebra. NumPy builds static basis/cut maps and random inputs only.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
import tensorcircuit as tc


def sector_indices(L):
    if L < 2 or L % 2:
        raise ValueError("L must be positive, even, and at least 2")
    return np.array([i for i in range(2**L) if bin(i).count("1") == L // 2])


def hamiltonian(L, fields, jz=0.5, jxy=1.0, omit_bonds=()):
    """Eq. (1): S=Pauli/2, OBC, restricted to total Sz=0."""
    terms, weights = [], []
    for i in range(L - 1):
        if i in omit_bonds:
            continue
        for code, weight in [(1, jxy / 4), (2, jxy / 4), (3, jz / 4)]:
            if weight:
                p = [0] * L
                p[i] = p[i + 1] = code
                terms.append(p)
                weights.append(weight)
    for i, field in enumerate(fields):
        p = [0] * L
        p[i] = 3
        terms.append(p)
        weights.append(field / 2)
    full = tc.quantum.PauliStringSum2COO(terms, weights, numpy=True).tocsr()
    idx = sector_indices(L)
    return tc.backend.convert_to_tensor(full[idx][:, idx].real.toarray())


class Evolution:
    def __init__(self, H):
        self.e, self.v = tc.backend.eigh(H)

    def at(self, psi, times):
        """Yield states without retaining all time snapshots in memory."""
        K = tc.backend
        coeff = K.transpose(K.conj(self.v)) @ psi
        for t in times:
            # Phase reduction cannot remove double-precision eigenvalue error.
            angle = K.mod(
                K.cast(self.e, "longdouble") * K.cast(t, "longdouble"),
                2 * K.acos(K.cast(-1, "longdouble")),
            )
            phase = K.exp(-1j * K.cast(angle, "float64"))
            state = self.v @ (
                phase[:, None] * coeff if coeff.ndim == 2 else phase * coeff
            )
            norm = K.sqrt(K.sum(K.abs(state) ** 2, axis=0))
            yield state / norm


class Floquet:
    def __init__(self, L, fields, xy_evolution=None):
        xy = xy_evolution or Evolution(hamiltonian(L, np.zeros(L), jz=0))
        Uxy = (xy.v * tc.backend.exp(-0.4j * xy.e)) @ tc.backend.transpose(xy.v)
        diagonal = tc.backend.diagonal(hamiltonian(L, fields, jz=1, jxy=0))
        F = tc.backend.exp(-1j * diagonal)[:, None] * Uxy
        triangular, self.v = tc.backend.schur(F, output="complex")
        diagonal = tc.backend.diagonal(triangular)
        self.e = -tc.backend.atan2(tc.backend.imag(diagonal), tc.backend.real(diagonal))

    at = Evolution.at


@lru_cache(maxsize=32)
def gate(alpha, beta):
    """Eq. (2) using TC rotations exp(-i theta Pauli-product / 2)."""
    c = tc.Circuit(2)
    c.rxx(0, 1, theta=alpha / 2)
    c.ryy(0, 1, theta=alpha / 2)
    c.rzz(0, 1, theta=beta / 2)
    return c.matrix()


class SectorGates:
    """Exact scatter application of TC gate matrices in the conserved sector.

    Equivalent to tc.Circuit(L, inputs=full_state).any(...); validated against
    that path. Avoids repeated full-Hilbert-space tensor construction.
    """

    def __init__(self, L):
        idx = sector_indices(L)
        inverse = np.full(2**L, -1, dtype=int)
        inverse[idx] = np.arange(len(idx))
        self.maps = []
        for bond in range(L - 1):
            p, q = L - 1 - bond, L - 2 - bond
            code = ((idx >> p) & 1) * 2 + ((idx >> q) & 1)
            opposite = (code == 1) | (code == 2)
            partner = np.arange(len(idx))
            partner[opposite] = inverse[idx[opposite] ^ (1 << p) ^ (1 << q)]
            self.maps.append((code, partner, opposite))

    def apply(self, state, U, bond):
        code, partner, opposite = self.maps[bond]
        diag = U[code, code]
        off = U[code, 3 - code] * opposite
        if state.ndim == 2:
            diag, off = diag[:, None], off[:, None]
        return diag * state + off * state[partner]


class Entropy:
    def __init__(self, L):
        self.L = L
        self.idx = sector_indices(L)
        self.half = tuple(range(L // 2))

    @lru_cache(maxsize=32)
    def blocks(self, cut):
        rest = tuple(i for i in range(self.L) if i not in cut)

        def labels(sites):
            out = np.zeros(len(self.idx), dtype=int)
            for i in sites:
                out = out * 2 + ((self.idx >> (self.L - 1 - i)) & 1)
            return out

        a, b = labels(cut), labels(rest)
        counts = np.array([bin(int(x)).count("1") for x in a])
        result = []
        for n in range(len(cut) + 1):
            mask = np.flatnonzero(counts == n)
            if mask.size == 0:
                continue
            aa, ai = np.unique(a[mask], return_inverse=True)
            bb, bi = np.unique(b[mask], return_inverse=True)
            result.append((mask, ai, bi, (len(aa), len(bb))))
        return result

    def value(self, state, cut=None):
        cut = self.half if cut is None else tuple(cut)
        entropy = 0.0
        norm = tc.backend.sum(tc.backend.abs(state) ** 2)
        for mask, ai, bi, shape in self.blocks(cut):
            block = tc.backend.scatter(
                tc.backend.zeros(shape), np.column_stack((ai, bi)), state[mask]
            )
            _, singular, _, _ = tc.backend.svd(block, pivot_axis=1)
            p = tc.backend.abs(singular) ** 2 / norm
            safe = tc.backend.where(p > 0, p, tc.backend.ones_like(p))
            entropy -= tc.backend.sum(p * tc.backend.log(safe)) / np.log(2)
        return float(entropy)

    def cuts(self, maximum=None, rng=None):
        # Exactly one representative of A/complement: A contains site 0.
        all_cuts = [(0,) + x for x in combinations(range(1, self.L), self.L // 2 - 1)]
        if maximum is not None and maximum < len(all_cuts):
            ids = rng.choice(len(all_cuts), maximum, replace=False)
            return [all_cuts[i] for i in ids]
        return all_cuts

    def average(self, state, cuts):
        return sum(self.value(state, c) for c in cuts) / len(cuts)

    def average_many(self, states, cuts):
        total = tc.backend.zeros((states.shape[1],), dtype="float64")
        for cut in cuts:
            total = total + tc.backend.convert_to_tensor(
                [self.value(states[:, i], cut) for i in range(states.shape[1])]
            )
        return total / len(cuts)
