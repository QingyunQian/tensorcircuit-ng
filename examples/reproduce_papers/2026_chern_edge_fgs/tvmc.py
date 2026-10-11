"""
Born sampling and gauge-projected PEPS-tVMC, following PRX Quantum 7, 033035.

The standard SR equation is Eq. (13); minSR uses the published uncentered
form, Eq. (17), with the real-time factor -i restored as in Eq. (18).
"""

import math

import tensorcircuit as tc

if __package__:
    from .sampling import sequential_sweep
else:
    from sampling import sequential_sweep


def sr_moments_solve(metric, force, gauge_basis, regulator=1e-8):
    """Project sufficient statistics using thin gauge factors, then Cholesky."""
    K = tc.backend
    metric = (metric + K.adjoint(metric)) / 2
    coupled = metric @ gauge_basis
    reduced = K.adjoint(gauge_basis) @ coupled
    correction = (coupled - 0.5 * gauge_basis @ reduced) @ K.adjoint(gauge_basis)
    projected = metric - correction - K.adjoint(correction)
    identity = K.eye(metric.shape[0], dtype="complex128")
    system = projected + regulator * identity + gauge_basis @ K.adjoint(gauge_basis)
    rhs = force - gauge_basis @ (K.adjoint(gauge_basis) @ force)
    velocity = K.solve(system, rhs, assume_a="pos")
    velocity = velocity - gauge_basis @ (K.adjoint(gauge_basis) @ velocity)
    # Reconstruct S v from projected S and S U; retain no second full S.
    residual_force = (
        projected @ velocity + gauge_basis @ (K.adjoint(coupled) @ velocity) - force
    )
    return velocity, K.norm(residual_force) ** 2 / K.norm(force) ** 2


def sr_solve(
    scores, energies, weights, projector=None, regulator=1e-8, gauge_basis=None
):
    """Dense sampled reference, also used to validate streamed sufficient statistics."""
    K = tc.backend
    mean_score = K.sum(weights[:, None] * scores, axis=0)
    mean_energy = K.sum(weights * energies)
    centered = scores - mean_score
    metric = K.adjoint(centered) @ (weights[:, None] * centered)
    force = K.adjoint(centered) @ (weights * (energies - mean_energy))
    if gauge_basis is not None:
        velocity, residual = sr_moments_solve(metric, force, gauge_basis, regulator)
    else:
        if projector is None:
            raise ValueError(
                "Provide an analytic gauge basis or a reference projector."
            )
        projected = projector @ metric @ projector
        projected = (projected + K.adjoint(projected)) / 2
        rhs = projector @ force
        identity = K.eye(metric.shape[0], dtype="complex128")
        system = projected + regulator * projector + identity - projector
        velocity = projector @ K.solve(system, rhs, assume_a="pos")
        residual = K.norm(metric @ velocity - force) ** 2 / K.norm(force) ** 2
    return velocity, mean_energy, residual


def sr_cg_solve(scores, energies, weights, regulator=1e-8, tolerance=1e-8, maxiter=512):
    """
    Centered ridge SR with matrix-free Fisher products and zero-start CG.

    Every iterate lies in the centered score row space. Exact gauge-null
    directions are therefore excluded implicitly, without constructing a gauge
    basis or projector. Finite-cap hole scores need not have exact gauge nulls;
    this is ridge SR on those approximate scores, not an analytic gauge fix.
    Storage is O(N P), each iteration costs O(N P), and no P x P or N x N
    metric is formed. Return the true relative linear-system residual as well
    as the unregularized squared force residual used by the other solvers.
    """
    if regulator <= 0 or tolerance <= 0 or maxiter < 1:
        raise ValueError("CG requires positive regulator, tolerance and maxiter.")
    K = tc.backend
    mean_energy = K.sum(weights * energies)
    centered = scores - K.sum(weights[:, None] * scores, axis=0)
    force = K.adjoint(centered) @ (weights * (energies - mean_energy))

    def fisher(vector):
        return K.adjoint(centered) @ (weights * (centered @ vector))

    def inner(vector):
        return K.real(K.sum(K.conj(vector) * vector))

    norm2 = inner(force)
    denominator = K.where(norm2 > 0, norm2, 1.0)

    def step(carry, _):
        vector, residual, direction, squared, iterations = carry

        def update():
            product = fisher(direction) + regulator * direction
            alpha = squared / K.real(K.sum(K.conj(direction) * product))
            next_residual = residual - alpha * product
            next_squared = inner(next_residual)
            return (
                vector + alpha * direction,
                next_residual,
                next_residual + (next_squared / squared) * direction,
                next_squared,
                iterations + 1,
            )

        return K.cond(squared > tolerance**2 * norm2, update, lambda: carry)

    velocity, _, _, _, iterations = K.scan(
        step,
        K.arange(maxiter),
        (
            K.zeros(force.shape, dtype="complex128"),
            force,
            force,
            norm2,
            K.cast(0, "int64"),
        ),
    )
    unregularized = fisher(velocity) - force
    linear_residual = K.sqrt(inner(unregularized + regulator * velocity) / denominator)
    return (
        velocity,
        mean_energy,
        inner(unregularized) / denominator,
        linear_residual,
        iterations,
    )


def minsr_solve(scores, energies, weights, regulator=1e-8):
    """Published uncentered minSR; use only in the underdetermined regime."""
    K = tc.backend
    root = K.sqrt(weights)
    design = root[:, None] * scores
    rhs = root * energies
    metric = design @ K.adjoint(design)
    metric = (metric + K.adjoint(metric)) / 2
    y = K.solve(
        metric + regulator * K.eye(metric.shape[0], dtype="complex128"),
        rhs,
        assume_a="pos",
    )
    velocity = K.adjoint(design) @ y
    centered = scores - K.sum(weights[:, None] * scores, axis=0)
    local_residual = centered @ velocity - (energies - K.sum(weights * energies))
    force = K.adjoint(centered) @ (weights * energies)
    residual = (
        K.norm(K.adjoint(centered) @ (weights * local_residual)) ** 2
        / K.norm(force) ** 2
    )
    return velocity, K.sum(weights * energies), residual


class MonteCarlo:
    """Number-conserving chains with exact or sequential boundary contractions."""

    def __init__(
        self,
        problem,
        chains=64,
        draws=32,
        sweeps=2,
        solver="sr",
        regulator=1e-8,
        contraction_batch_size=256,
        cg_tolerance=1e-8,
        cg_maxiter=512,
    ):
        if (
            chains < 2
            or min(draws, sweeps) < 1
            or solver not in ("sr", "minsr", "dense")
        ):
            raise ValueError(
                "At least two chains, positive draws/sweeps, and sr/minsr/dense are required."
            )
        self.problem, self.peps = problem, problem.peps
        self.chains, self.draws, self.sweeps = chains, draws, sweeps
        self.solver, self.regulator = solver, regulator
        if regulator <= 0 or cg_tolerance <= 0 or cg_maxiter < 1:
            raise ValueError("Positive regulator and CG controls are required.")
        self.cg_tolerance, self.cg_maxiter = cg_tolerance, cg_maxiter
        if contraction_batch_size < 1:
            raise ValueError("The contraction batch size must be positive.")
        self.contraction_batch_size = contraction_batch_size
        if solver == "minsr" and chains * draws >= self.peps.nparams:
            raise ValueError(
                "minSR requires fewer sampled rows than parameters; use SR."
            )
        self.capacity = min(
            chains * draws, math.comb(problem.nsites, problem.particles)
        )
        self.compress_samples = problem.nsites <= 20
        if self.compress_samples:
            self.powers = tc.backend.convert_to_tensor(
                [2**i for i in range(problem.nsites)]
            )

    def initial_chains(self, key):
        K = tc.backend
        random_values = K.stateful_randu(key, (self.chains, self.problem.nsites))
        return K.cast(
            K.argsort(random_values, axis=1) < self.problem.particles, "int64"
        )

    def advance(self, theta, chains, key, steps):
        """One transition satisfies detailed balance for the actual amplitude function."""
        K = tc.backend
        logp = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, chains)))

        def step(carry, _):
            state, current, rng, accepted = carry
            rng, draw = K.random_split(rng)
            random_values = K.stateful_randu(draw, (self.chains, 2))
            bond = K.cast(random_values[:, 0] * self.problem.left.shape[0], "int64")
            permutations = self.problem.permutations[bond]
            proposal = K.vmap(lambda s, p: s[p], vectorized_argnums=(0, 1))(
                state, permutations
            )
            proposed = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, proposal)))
            changed = K.sum(K.abs(proposal - state), axis=1) > 0
            accept = K.log(random_values[:, 1]) < proposed - current
            return (
                K.where(accept[:, None], proposal, state),
                K.where(accept, proposed, current),
                rng,
                accepted + K.sum(K.cast(accept & changed, "float64")),
            )

        chains, _, key, accepted = K.scan(
            step, K.arange(steps), (chains, logp, key, K.cast(0.0, "float64"))
        )
        return chains, key, accepted / (steps * self.chains)

    def sweep(self, theta, chains, key, sweeps):
        """Sequential conditional exchanges; each sweep visits every bond once."""
        K = tc.backend
        update = K.vmap(
            lambda p, s, u: sequential_sweep(self.peps, p, s, u),
            vectorized_argnums=(1, 2),
        )

        def step(carry, _):
            states, rng, accepted = carry
            rng, draw = K.random_split(rng)
            uniforms = K.stateful_randu(
                draw, (self.chains, 2, self.peps.rows, self.peps.columns)
            )
            states, rate = update(theta, states, uniforms)
            return states, rng, accepted + K.mean(rate)

        states, key, accepted = K.scan(
            step, K.arange(sweeps), (chains, key, K.cast(0.0, "float64"))
        )
        return states, key, accepted / sweeps

    def sample(self, theta, chains, key):
        K = tc.backend
        samples = K.zeros((self.draws, self.chains, self.problem.nsites), dtype="int64")

        def draw(carry, i):
            state, rng, buffer, accepted = carry
            if self.peps.boundary_dim is None:
                state, rng, rate = self.advance(
                    theta, state, rng, self.sweeps * self.problem.left.shape[0]
                )
            else:
                state, rng, rate = self.sweep(theta, state, rng, self.sweeps)
            buffer = K.scatter(buffer, K.reshape(i, (1, 1)), state[None])
            return state, rng, buffer, accepted + rate

        chains, key, samples, accepted = K.scan(
            draw, K.arange(self.draws), (chains, key, samples, K.cast(0.0, "float64"))
        )
        return (
            K.reshape(samples, (-1, self.problem.nsites)),
            chains,
            key,
            accepted / self.draws,
        )

    def rows(self, samples):
        """Coalesce only configurations actually sampled; never enumerate the basis."""
        K = tc.backend
        if self.compress_samples:
            labels = samples @ self.powers
            labels, counts = K.unique_with_counts(
                labels, size=self.capacity, fill_value=labels[0]
            )
            configurations = K.mod(labels[:, None] // self.powers[None, :], 2)
            return configurations, counts / K.sum(counts)
        return samples, K.ones((samples.shape[0],), dtype="float64") / samples.shape[0]

    def sampled_moments(self, theta, configurations, potential=0.0):
        """Bound contraction intermediates while retaining all sampled SR rows."""
        K = tc.backend

        def evaluate(states):
            _, scores = self.peps.batch_scores(theta, states)
            return scores, self.problem.local_batch(theta, states, potential)

        count, batch = configurations.shape[0], self.contraction_batch_size
        if count <= batch:
            return evaluate(configurations)
        padding = (-count) % batch
        states = K.concat([configurations, K.tile(configurations[:1], (padding, 1))])
        blocks = K.reshape(states, (-1, batch, self.problem.nsites))
        _, (scores, energies) = K.jaxy_scan(
            lambda carry, block: (carry, evaluate(block)), None, blocks
        )
        return (
            K.reshape(scores, (-1, self.peps.nparams))[:count],
            K.reshape(energies, (-1,))[:count],
        )

    def estimate(self, theta, chains, key, potential=0.0):
        K = tc.backend
        samples, chains, key, accepted = self.sample(theta, chains, key)
        configurations, weights = self.rows(samples)
        if self.solver == "dense":
            metric, force, energy, energies = self.sr_statistics(
                theta, configurations, weights, potential
            )
            direction, residual = sr_moments_solve(
                metric, force, self.peps.gauge_basis(theta), self.regulator
            )
        else:
            scores, energies = self.sampled_moments(theta, configurations, potential)
            if self.solver == "minsr":
                direction, energy, residual = minsr_solve(
                    scores, energies, weights, self.regulator
                )
            else:
                direction, energy, residual, linear_residual, _ = sr_cg_solve(
                    scores,
                    energies,
                    weights,
                    self.regulator,
                    self.cg_tolerance,
                    self.cg_maxiter,
                )
                # A failed iterative solve must not silently advance a trajectory.
                direction = K.where(
                    linear_residual <= self.cg_tolerance * 10,
                    direction,
                    K.cast(float("nan"), "complex128"),
                )
        blocks = K.mean(
            K.reshape(
                K.cast(samples, "float64"),
                (self.draws, self.chains, self.problem.nsites),
            ),
            axis=0,
        )
        density = K.mean(blocks, axis=0)
        error = K.sqrt(
            K.sum((blocks - density) ** 2, axis=0) / (self.chains * (self.chains - 1))
        )
        variance = K.real(K.sum(weights * K.abs(energies - energy) ** 2))
        diagnostics = K.concat(
            [
                K.real(energy)[None],
                variance[None],
                residual[None],
                accepted[None],
                density,
                error,
            ]
        )
        return direction, diagnostics, chains, key

    def sr_statistics(self, theta, configurations, weights, potential=0.0):
        """
        Stream weighted covariance blocks without storing the full score matrix.

        Weighted parallel-variance merging avoids subtracting two large raw
        second moments. Padding has zero weight and does not change the estimator.
        """
        K = tc.backend
        count = configurations.shape[0]
        batch = min(count, self.contraction_batch_size)
        padding = (-count) % batch
        states = K.concat([configurations, K.tile(configurations[:1], (padding, 1))])
        probabilities = K.concat([weights, K.zeros((padding,), dtype="float64")])
        blocks = (
            K.reshape(states, (-1, batch, self.problem.nsites)),
            K.reshape(probabilities, (-1, batch)),
        )

        def accumulate(carry, block):
            mass, mean, mean_e, metric, force = carry
            states, probability = block
            _, scores = self.peps.batch_scores(theta, states)
            energies = self.problem.local_batch(theta, states, potential)
            block_mass = K.sum(probability)
            denominator = K.where(block_mass > 0, block_mass, 1.0)
            block_mean = K.sum(probability[:, None] * scores, axis=0) / denominator
            block_energy = K.sum(probability * energies) / denominator
            centered = scores - block_mean
            centered_energy = energies - block_energy
            total = mass + block_mass
            total_denominator = K.where(total > 0, total, 1.0)
            fraction = block_mass / total_denominator
            correction = mass * fraction
            difference, difference_e = block_mean - mean, block_energy - mean_e
            metric = (
                metric
                + K.adjoint(centered) @ (probability[:, None] * centered)
                + correction * K.conj(difference)[:, None] * difference[None, :]
            )
            force = (
                force
                + K.adjoint(centered) @ (probability * centered_energy)
                + correction * K.conj(difference) * difference_e
            )
            return (
                total,
                mean + fraction * difference,
                mean_e + fraction * difference_e,
                metric,
                force,
            ), energies

        zero = K.zeros((self.peps.nparams,), dtype="complex128")
        initial = (
            K.cast(0.0, "float64"),
            zero,
            K.cast(0.0, "complex128"),
            K.zeros((self.peps.nparams, self.peps.nparams), dtype="complex128"),
            zero,
        )
        (mass, _, energy, metric, force), energies = K.jaxy_scan(
            accumulate, initial, blocks
        )
        return metric / mass, force / mass, energy, K.reshape(energies, (-1,))[:count]

    def rk4_step(self, theta, chains, key, dt, potential=0.0, imaginary=False):
        """Re-equilibrate chains at each RK stage; carry RNG state between stages."""
        K = tc.backend
        factor = -1.0 if imaginary else -1j
        offsets = K.convert_to_tensor([0.0, 0.5, 0.5, 1.0])
        weights = K.convert_to_tensor([1.0, 2.0, 2.0, 1.0])

        def stage(carry, index):
            previous, total, states, rng, first = carry
            direction, diagnostics, states, rng = self.estimate(
                theta + factor * dt * offsets[index] * previous, states, rng, potential
            )
            return (
                direction,
                total + weights[index] * direction,
                states,
                rng,
                K.where(index == 0, diagnostics, first),
            )

        zero = K.zeros(theta.shape, dtype="complex128")
        _, total, chains, key, diagnostics = K.scan(
            stage,
            K.arange(4),
            (
                zero,
                zero,
                chains,
                key,
                K.zeros((4 + 2 * self.peps.nsites,), dtype="float64"),
            ),
        )
        theta = self.peps.normalize(theta + factor * dt * total / 6)
        return theta, chains, key, diagnostics

    def trajectory(
        self,
        theta,
        chains,
        key,
        dt,
        steps,
        potential=0.0,
        imaginary=False,
        prepare_euler=False,
    ):
        """A complete fixed-step trajectory executes within a backend scan."""
        if prepare_euler and not imaginary:
            raise ValueError("Euler updates are only used for VMC preparation.")
        K = tc.backend
        history = K.zeros((steps, 4 + 2 * self.problem.nsites), dtype="float64")

        def step(carry, index):
            params, states, rng, buffer = carry
            if prepare_euler:
                direction, diagnostics, states, rng = self.estimate(
                    params, states, rng, potential
                )
                params = self.peps.normalize(params - dt * direction)
            else:
                params, states, rng, diagnostics = self.rk4_step(
                    params, states, rng, dt, potential, imaginary
                )
            buffer = K.scatter(buffer, K.reshape(index, (1, 1)), diagnostics[None])
            return params, states, rng, buffer

        return K.scan(step, K.arange(steps), (theta, chains, key, history))
