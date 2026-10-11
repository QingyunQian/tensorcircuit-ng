"""
Number-projected fermionic PEPS in the swap-gate formulation.

The physical order is i = y * columns + x, with y increasing upward.
Physical legs leave toward the upper left; a leg at (x, y) crosses the
upward virtual bonds to its left. See Eqs. (2), (3), (8) of
https://arxiv.org/abs/2506.20106. Virtual indices alternate even/odd parity.
"""

import cmath
import itertools
import json
from pathlib import Path
import time
import math

import tencirpauli as tcp
import tensorcircuit as tc

if __package__:
    from ...peps_boundary_mps import apply_grid_row_dmrg, peps_partition_function
else:
    from peps_boundary_mps import apply_grid_row_dmrg, peps_partition_function


class FermionPEPS:
    """Open, even-parity PEPS; pack only parity-allowed tensor entries."""

    def __init__(
        self,
        rows,
        columns,
        bond_dim=2,
        boundary_dim=None,
        exact_optimizer="omeco-8-48",
        path_file=None,
    ):
        if rows < 2 or columns < 2 or bond_dim < 2 or bond_dim % 2:
            raise ValueError("Use a rectangle of at least 2 x 2 and an even D >= 2.")
        if boundary_dim is not None and boundary_dim < 1:
            raise ValueError("The boundary-MPS cap must be positive.")
        self.rows, self.columns = rows, columns
        self.nsites = rows * columns
        self.bond_dim, self.boundary_dim = bond_dim, boundary_dim
        self.contraction_key = tc.backend.get_random_state(0)
        self.shapes, self.offsets, self.allowed, self.coordinates = [], [0], [], []
        for y in range(rows):
            for x in range(columns):
                shape = (
                    2,
                    bond_dim if x else 1,
                    bond_dim if x + 1 < columns else 1,
                    bond_dim if y else 1,
                    bond_dim if y + 1 < rows else 1,
                )
                coordinates = []
                for virtual in itertools.product(*(range(d) for d in shape[1:])):
                    coordinates.append((sum(virtual) % 2,) + virtual)
                indices = []
                for coordinate in coordinates:
                    index = 0
                    for k, d in zip(coordinate, shape):
                        index = index * d + k
                    indices.append(index)
                self.shapes.append(shape)
                self.coordinates.append(coordinates)
                self.allowed.append(tc.backend.convert_to_tensor(indices))
                self.offsets.append(self.offsets[-1] + len(indices))
        self.nparams = self.offsets[-1]
        self.bonds = []
        for y in range(rows):
            for x in range(columns):
                i = y * columns + x
                if x + 1 < columns:
                    self.bonds.append((i, i + 1, 2, 1))
                if y + 1 < rows:
                    self.bonds.append((i, i + columns, 4, 3))
        if boundary_dim is None:
            self.prepare_exact_path(exact_optimizer, path_file)
        self.batch_amplitude = tc.backend.vmap(self.amplitude, vectorized_argnums=1)
        self.value_derivative = tc.backend.value_and_grad(
            lambda theta, occupation: tc.backend.real(self.amplitude(theta, occupation))
        )
        self.batch_scores = tc.backend.vmap(self.scores, vectorized_argnums=1)
        self._make_gauge_indices()

    def _make_gauge_indices(self):
        """Sparse generators of parity-preserving virtual GL(D/2) x GL(D/2)."""
        output, source, columns = [], [], []
        lookups = [dict(zip(c, range(len(c)))) for c in self.coordinates]
        column = 0
        for i, j, ai, aj in self.bonds:
            for a in range(self.bond_dim):
                for b in range(self.bond_dim):
                    if (a - b) % 2:
                        continue
                    for site, axis, old, new, sign in (
                        (i, ai, a, b, 1),
                        (j, aj, b, a, -1),
                    ):
                        for k, coordinate in enumerate(self.coordinates[site]):
                            if coordinate[axis] == new:
                                changed = list(coordinate)
                                changed[axis] = old
                                output.append(self.offsets[site] + k)
                                source.append(
                                    self.offsets[site] + lookups[site][tuple(changed)]
                                )
                                columns.append((column, sign))
                    column += 1
        self.gauge_output = tc.backend.convert_to_tensor(output)
        self.gauge_source = tc.backend.convert_to_tensor(source)
        self.gauge_columns = tc.backend.convert_to_tensor([c for c, _ in columns])
        self.gauge_signs = tc.backend.convert_to_tensor([s for _, s in columns])
        self.ngenerators = column
        self.physical_index = tc.backend.convert_to_tensor(
            [c[0] for site in self.coordinates for c in site]
        )

    def random_parameters(self, key):
        """Generic complex tensors avoid the stationary product-state manifold."""
        K = tc.backend
        k1, k2 = K.random_split(key)
        theta = K.stateful_randn(k1, [self.nparams]) + 1j * K.stateful_randn(
            k2, [self.nparams]
        )
        return self.normalize(theta)

    def normalize(self, theta):
        """Remove irrelevant tensor scales without changing the physical state."""
        K = tc.backend
        return K.concat(
            [
                theta[a:b] / K.norm(theta[a:b])
                for a, b in zip(self.offsets[:-1], self.offsets[1:])
            ]
        )

    def tensors(self, theta):
        """Expand parity blocks to physical,left,right,down,up tensors."""
        K = tc.backend
        return [
            K.reshape(
                K.scatter(
                    K.zeros([math.prod(shape)], dtype="complex128"),
                    indices[:, None],
                    theta[a:b],
                ),
                shape,
            )
            for shape, indices, a, b in zip(
                self.shapes, self.allowed, self.offsets[:-1], self.offsets[1:]
            )
        ]

    def sliced_tensors(self, theta, occupation):
        """Absorb every sampled physical/virtual fermionic swap exactly once."""
        K = tc.backend
        grid = K.reshape(occupation, (self.rows, self.columns))
        right = K.sum(grid, axis=1)[:, None] - K.cumsum(grid, axis=1)
        tensors = self.tensors(theta)
        sliced = []
        for i, tensor in enumerate(tensors):
            y, x = divmod(i, self.columns)
            sign = 1 - 2 * K.mod(right[y, x] * K.arange(tensor.shape[-1]), 2)
            sliced.append(tensor[occupation[i]] * sign[None, None, None, :])
        return sliced

    def amplitude(self, theta, occupation):
        """
        Contract a sampled single-layer fermionic PEPS.

        boundary_dim=None uses a cached exact pairwise contraction path. Finite caps call
        TensorCircuit-NG's existing variational boundary-MPS contractor.
        """
        K = tc.backend
        tensors = self.sliced_tensors(theta, occupation)
        if self.boundary_dim is not None:
            return peps_partition_function(
                self.padded_grid(tensors),
                self.boundary_dim,
                self.contraction_key,
                num_sweeps=2,
            )
        values = [K.reshape(t, shape) for t, shape in zip(tensors, self.exact_shapes)]
        for a, b, axes in self.exact_steps:
            result = K.tensordot(values[a], values[b], axes)
            values = [v for i, v in enumerate(values) if i not in (a, b)]
            values.append(result)
        return K.reshape(values[0], ())

    def prepare_exact_path(self, optimizer="omeco-8-48", path_file=None):
        """
        Search once outside JIT, or load a topology-checked pairwise path.

        Exterior dimension-one legs are removed without approximation. Cached
        steps use only backend tensordot, so configurations, connected states and
        derivatives share one path, independently of the global contractor.
        A supplied path file is loaded if present, otherwise searched and saved.
        """
        labels = [[None] * 4 for _ in self.shapes]
        for bond, (i, j, ai, aj) in enumerate(self.bonds):
            labels[i][ai - 1] = bond
            labels[j][aj - 1] = bond
        inputs = [[ix for ix in site if ix is not None] for site in labels]
        shapes = [[self.bond_dim] * len(site) for site in inputs]
        topology = {"inputs": inputs, "shapes": shapes, "output": []}
        path_file = Path(path_file) if path_file is not None else None
        start = time.perf_counter()
        if path_file is not None and path_file.exists():
            record = json.loads(path_file.read_text())
            if record["topology"] != topology:
                raise ValueError("The saved contraction path has a different topology.")
            path = record["path"]
        else:
            finder = tc.get_contractor(optimizer).keywords["optimizer"]
            path = finder(
                inputs, [], {i: self.bond_dim for i in range(len(self.bonds))}
            )
            record = {
                "topology": topology,
                "optimizer": optimizer,
                "path": [list(pair) for pair in path],
            }
        self.exact_path_seconds = time.perf_counter() - start
        current = [list(site) for site in inputs]
        steps = []
        for pair in path:
            if (
                len(pair) != 2
                or len(set(pair)) != 2
                or any(i < 0 or i >= len(current) for i in pair)
            ):
                raise ValueError("Invalid pair in saved contraction path.")
            a, b = pair
            common = [ix for ix in current[a] if ix in current[b]]
            axes = (
                [current[a].index(ix) for ix in common],
                [current[b].index(ix) for ix in common],
            )
            remaining = [ix for ix in current[a] + current[b] if ix not in common]
            current = [site for i, site in enumerate(current) if i not in (a, b)]
            current.append(remaining)
            steps.append((a, b, axes))
        if current != [[]]:
            raise ValueError(
                "The contraction path does not reduce the network to a scalar."
            )
        self.exact_shapes, self.exact_steps = shapes, steps
        self.exact_path_record = record
        if path_file is not None and not path_file.exists():
            path_file.parent.mkdir(parents=True, exist_ok=True)
            path_file.write_text(json.dumps(record, indent=2) + "\n")

    def padded_grid(self, tensors):
        """Adapt sliced tensors to the existing contractor's uniform grid layout."""
        K = tc.backend
        padded = []
        for tensor in tensors:
            values = K.transpose(tensor, (2, 3, 0, 1))
            coordinates = list(itertools.product(*(range(d) for d in values.shape)))
            grid = K.scatter(
                K.zeros((self.bond_dim,) * 4, dtype="complex128"),
                K.convert_to_tensor(coordinates),
                K.reshape(values, (-1,)),
            )
            padded.append(grid)
        return K.reshape(
            K.stack(padded), (self.rows, self.columns) + (self.bond_dim,) * 4
        )

    def boundary_environments(self, theta, occupation):
        """Reuse the library's row compression for upper and lower environments."""
        K = tc.backend
        sliced = self.sliced_tensors(theta, occupation)
        grid = self.padded_grid(sliced)
        chi, D = self.boundary_dim, self.bond_dim
        initial = K.scatter(
            K.zeros((chi, D, chi), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        initial = K.stack([initial] * self.columns)
        lower, upper = [initial], [initial]
        key = self.contraction_key
        for y in range(self.rows - 1):
            key, draw = K.random_split(key)
            value, _, _ = apply_grid_row_dmrg(lower[-1], grid[y], chi, draw)
            lower.append(value)
            value, _, _ = apply_grid_row_dmrg(
                upper[-1],
                K.transpose(grid[self.rows - 1 - y], (0, 2, 1, 3, 4)),
                chi,
                draw,
            )
            upper.append(value)
        upper = upper[::-1]
        return sliced, grid, lower, upper

    def boundary_scores(self, theta, occupation):
        """Hole environments, Eq. (12) of arXiv:2506.20106; no AD through compression."""
        K = tc.backend
        sliced, grid, lower, upper = self.boundary_environments(theta, occupation)
        chi, D = self.boundary_dim, self.bond_dim
        endpoint = K.scatter(
            K.zeros((chi, chi, D), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        scores = []
        for y in range(self.rows):
            left = [endpoint]
            right = [endpoint]
            for x in range(self.columns):
                left.append(
                    K.einsum(
                        "abl,adA,buB,dulr->ABr",
                        left[-1],
                        lower[y][x],
                        upper[y][x],
                        grid[y, x],
                    )
                )
                k = self.columns - 1 - x
                right.append(
                    K.einsum(
                        "ABr,adA,buB,dulr->abl",
                        right[-1],
                        lower[y][k],
                        upper[y][k],
                        grid[y, k],
                    )
                )
            right = right[::-1]
            for x in range(self.columns):
                i = y * self.columns + x
                hole = K.einsum(
                    "abl,ABr,adA,buB->lrdu",
                    left[x],
                    right[x + 1],
                    lower[y][x],
                    upper[y][x],
                )
                shape = self.shapes[i]
                hole = hole[: shape[1], : shape[2], : shape[3], : shape[4]]
                denominator = K.sum(hole * sliced[i])
                sign = 1 - 2 * K.mod(
                    K.sum(occupation[i + 1 : (y + 1) * self.columns])
                    * K.arange(shape[-1]),
                    2,
                )
                derivative = (
                    K.onehot(occupation[i], 2)[:, None, None, None, None]
                    * (hole * sign[None, None, None, :] / denominator)[None]
                )
                scores.append(K.reshape(derivative, (-1,))[self.allowed[i]])
        return self.amplitude(theta, occupation), K.concat(scores)

    def boundary_hopping_ratios(self, theta, occupation):
        """
        Nearest-neighbor ratios with shared one- and two-row environments.

        A vertical exchange changes the physical/virtual swap signs at every
        site to its left in both rows. A second left environment carries that
        string; dropping it would produce incorrect fermion amplitudes.
        Ratios converge to the exact contractor as the boundary cap increases.
        """
        K = tc.backend
        _, grid, lower, upper = self.boundary_environments(theta, occupation)
        shape = (self.rows, self.columns)
        occupations = K.reshape(occupation, shape)
        right_count = K.sum(occupations, axis=1)[:, None] - K.cumsum(
            occupations, axis=1
        )
        flipped = []
        for i, tensor in enumerate(self.tensors(theta)):
            y, x = divmod(i, self.columns)
            sign = 1 - 2 * K.mod(right_count[y, x] * K.arange(tensor.shape[-1]), 2)
            flipped.append(tensor[1 - occupation[i]] * sign[None, None, None, :])
        flipped = self.padded_grid(flipped)
        chi, D = self.boundary_dim, self.bond_dim
        parity = 1 - 2 * K.mod(K.arange(D), 2)
        swap = parity[None, :, None, None]
        endpoint = K.scatter(
            K.zeros((chi, chi, D), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        horizontal, vertical = [], []
        for y in range(self.rows):
            right = [endpoint]
            for x in range(self.columns - 1, -1, -1):
                right.append(
                    K.einsum(
                        "ABr,adA,buB,dulr->abl",
                        right[-1],
                        lower[y][x],
                        upper[y][x],
                        grid[y, x],
                    )
                )
            right = right[::-1]
            denominator = K.sum(endpoint * right[0])
            left = endpoint
            for x in range(self.columns - 1):
                proposal = K.einsum(
                    "abl,adA,buB,dulr->ABr",
                    left,
                    lower[y][x],
                    upper[y][x],
                    flipped[y, x] * swap,
                )
                proposal = K.einsum(
                    "abl,adA,buB,dulr->ABr",
                    proposal,
                    lower[y][x + 1],
                    upper[y][x + 1],
                    flipped[y, x + 1],
                )
                horizontal.append(K.sum(proposal * right[x + 2]) / denominator)
                left = K.einsum(
                    "abl,adA,buB,dulr->ABr", left, lower[y][x], upper[y][x], grid[y, x]
                )
        endpoint = K.scatter(
            K.zeros((chi, chi, D, D), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        for y in range(self.rows - 1):
            right = [endpoint]
            for x in range(self.columns - 1, -1, -1):
                right.append(
                    K.einsum(
                        "ABrs,adA,buB,dvlr,vums->ablm",
                        right[-1],
                        lower[y][x],
                        upper[y + 1][x],
                        grid[y, x],
                        grid[y + 1, x],
                    )
                )
            right = right[::-1]
            denominator = K.sum(endpoint * right[0])
            left_string = endpoint
            for x in range(self.columns):
                proposal = K.einsum(
                    "ablm,adA,buB,dvlr,vums->ABrs",
                    left_string,
                    lower[y][x],
                    upper[y + 1][x],
                    flipped[y, x],
                    flipped[y + 1, x],
                )
                vertical.append(K.sum(proposal * right[x + 1]) / denominator)
                left_string = K.einsum(
                    "ablm,adA,buB,dvlr,vums->ABrs",
                    left_string,
                    lower[y][x],
                    upper[y + 1][x],
                    grid[y, x] * swap,
                    grid[y + 1, x] * swap,
                )
        ratios = []
        for i, j, _, _ in self.bonds:
            y, x = divmod(i, self.columns)
            value = (
                horizontal[y * (self.columns - 1) + x]
                if j == i + 1
                else vertical[y * self.columns + x]
            )
            ratios.append(K.where(occupation[i] != occupation[j], value, 0.0))
        return K.stack(ratios)

    def scores(self, theta, occupation):
        """Holomorphic derivative of log amplitude, with JAX's complex convention."""
        if self.boundary_dim is not None:
            return self.boundary_scores(theta, occupation)
        amplitude = self.amplitude(theta, occupation)
        _, derivative = self.value_derivative(theta, occupation)
        return amplitude, derivative / amplitude

    def gauge_vectors(self, theta):
        """Include virtual gauge, global scale, and fixed-number scale directions."""
        K = tc.backend
        indices = K.stack([self.gauge_output, self.gauge_columns], axis=1)
        vectors = K.scatter(
            K.zeros((self.nparams, self.ngenerators), dtype="complex128"),
            indices,
            theta[self.gauge_source] * self.gauge_signs,
        )
        return K.concat(
            [vectors, theta[:, None], (theta * self.physical_index)[:, None]], axis=1
        )

    def gauge_basis(self, theta):
        """Orthonormal gauge columns, with dependent columns set to zero."""
        K = tc.backend
        vectors = self.gauge_vectors(theta)
        u, singular, _, _ = K.svd(vectors)
        active = K.cast(K.real(singular) > K.real(singular[0]) * 1e-10, "complex128")
        return u * active

    def gauge_projector(self, theta):
        """Dense reference projector for independent small-system validation."""
        K = tc.backend
        basis = self.gauge_basis(theta)
        return K.eye(self.nparams, dtype="complex128") - basis @ K.adjoint(basis)


class Hofstadter:
    """Open spinless-fermion hopping; the published negative Peierls exponent."""

    def __init__(self, peps, particles):
        if particles % 2 or not 0 < particles < peps.nsites:
            raise ValueError("This even-parity example requires an even 0 < N < sites.")
        self.peps, self.particles = peps, particles
        self.nsites = peps.nsites
        self.corner = (peps.rows - 1) * peps.columns
        K = tc.backend
        pairs = [(i, j) for i, j, _, _ in peps.bonds]
        self.left = K.convert_to_tensor([i for i, _ in pairs])
        self.right = K.convert_to_tensor([j for _, j in pairs])
        phases, permutations, terms = [], [], []
        for i, j in pairs:
            phase = (
                -cmath.exp(-2j * math.pi * (i % peps.columns) / 3)
                if j - i == peps.columns
                else -1.0
            )
            phases.append(phase)
            terms.extend(
                [
                    (((i, "create"), (j, "annihilate")), phase),
                    (((j, "create"), (i, "annihilate")), complex(phase).conjugate()),
                ]
            )
            permutation = list(range(self.nsites))
            permutation[i], permutation[j] = permutation[j], permutation[i]
            permutations.append(permutation)
        self.hopping = K.stack([K.cast(h, "complex128") for h in phases])
        self.permutations = K.convert_to_tensor(permutations)
        self.operator = tcp.FermionOperator.from_terms(self.nsites, terms)
        self.pin_operator = tcp.FermionOperator.from_terms(
            self.nsites, [(((self.corner, "create"), (self.corner, "annihilate")), 1.0)]
        )
        self.pauli = self.operator.map_fermions()
        x_masks, z_masks = [], []
        for term in self.pauli.terms:
            codes = term.word.to_codes()
            x_masks.append(tuple(int(code in (1, 2)) for code in codes))
            z_masks.append(tuple(int(code in (2, 3)) for code in codes))
        unique = list(dict.fromkeys(x_masks))
        groups = [unique.index(mask) for mask in x_masks]
        self.flips = K.convert_to_tensor(unique)
        self.hopping_groups = K.convert_to_tensor(
            [
                unique.index(tuple(int(k in (i, j)) for k in range(self.nsites)))
                for i, j in pairs
            ]
        )
        self.z_masks = K.convert_to_tensor(z_masks)
        # Pauli terms avoid the full-state dimension limit of backend_mvp_plan.
        self.coefficients = K.convert_to_tensor(
            [
                term.coefficient * (-1j) ** sum(x * z for x, z in zip(xs, zs))
                for term, xs, zs in zip(self.pauli.terms, x_masks, z_masks)
            ]
        )
        self.grouping = K.transpose(K.onehot(K.convert_to_tensor(groups), len(unique)))
        self.local_batch = K.vmap(self.local_energy, vectorized_argnums=1)

    def local_energy(self, theta, occupation, potential=0.0):
        """Evaluate connected amplitudes using TenCirPauli's mapped Pauli terms."""
        K = tc.backend
        signs = 1 - 2 * K.mod(self.z_masks @ occupation, 2)
        matrix_elements = self.grouping @ (self.coefficients * signs)
        if self.peps.boundary_dim is not None:
            ratios = self.peps.boundary_hopping_ratios(theta, occupation)
            return (
                K.sum(matrix_elements[self.hopping_groups] * ratios)
                + potential * occupation[self.corner]
            )
        connected = K.mod(occupation[None, :] + self.flips, 2)
        amplitudes = self.peps.batch_amplitude(theta, connected)
        amplitudes = K.where(
            K.sum(connected, axis=1) == self.particles, amplitudes, 0.0
        )
        amplitude = self.peps.amplitude(theta, occupation)
        return (
            K.sum(matrix_elements * amplitudes) / amplitude
            + potential * occupation[self.corner]
        )

    def one_body(self, potential=0.0):
        """Single-particle Hamiltonian for the independent FGS reference only."""
        K = tc.backend
        h = K.scatter(
            K.zeros((self.nsites, self.nsites), dtype="complex128"),
            K.stack([self.left, self.right], axis=1),
            self.hopping,
        )
        h = h + K.adjoint(h)
        pin = K.onehot(self.corner, self.nsites)
        return h + potential * pin[:, None] * pin[None, :]
