"""Sequential exchange sampler, with shared fermionic PEPS environments."""

import tensorcircuit as tc

if __package__:
    from ...peps_boundary_mps import apply_grid_row_dmrg
else:
    from peps_boundary_mps import apply_grid_row_dmrg


def sequential_sweep(peps, theta, occupation, uniforms):
    """
    Visit every horizontal and vertical bond with shared row environments.

    For finite boundary caps these are approximate conditional probabilities;
    the sampled moments must be checked for convergence in the boundary cap.
    The exact-contraction limit preserves the fixed-number Born distribution.
    """
    K = tc.backend
    D, chi = peps.bond_dim, peps.boundary_dim
    tensors = peps.tensors(theta)
    physical = K.stack(
        [peps.padded_grid([a[p] for a in tensors]) for p in range(2)], axis=2
    )
    state = K.reshape(occupation, (peps.rows, peps.columns))
    grid = peps.padded_grid(peps.sliced_tensors(theta, occupation))
    initial = K.zeros((chi, D, chi), dtype="complex128")
    initial = K.scatter(
        initial, K.convert_to_tensor([[0, 0, 0]]), K.ones((1,), dtype="complex128")
    )
    initial = K.stack([initial] * peps.columns)
    parity = 1 - 2 * K.mod(K.arange(D), 2)
    swap = parity[None, :, None, None]

    def selected(values, y, x):
        count = K.sum(K.where(K.arange(peps.columns) > x, values[y], 0))
        sign = 1 - 2 * K.mod(count * K.arange(D), 2)
        return physical[y, x, values[y, x]] * sign[None, :, None, None]

    def left1(env, lo, up, a):
        return K.einsum("abl,adA,buB,dulr->ABr", env, lo, up, a)

    def right1(env, lo, up, a):
        return K.einsum("ABr,adA,buB,dulr->abl", env, lo, up, a)

    def left2(env, lo, up, a, b):
        return K.einsum("ablm,adA,buB,dvlr,vums->ABrs", env, lo, up, a, b)

    def right2(env, lo, up, a, b):
        return K.einsum("ABrs,adA,buB,dvlr,vums->ablm", env, lo, up, a, b)

    total = K.cast(0.0, "float64")
    for vertical in (False, True):
        upper = [initial]
        for y in range(peps.rows - 1, 0, -1):
            value, _, _ = apply_grid_row_dmrg(
                upper[-1],
                K.transpose(grid[y], (0, 2, 1, 3, 4)),
                chi,
                peps.contraction_key,
            )
            upper.append(value)
        upper = upper[::-1]
        lower = initial
        endpoint_shape = (chi, chi, D, D) if vertical else (chi, chi, D)
        endpoint = K.scatter(
            K.zeros(endpoint_shape, dtype="complex128"),
            K.convert_to_tensor([[0] * len(endpoint_shape)]),
            K.ones((1,), dtype="complex128"),
        )
        for y in range(peps.rows - int(vertical)):
            top = upper[y + int(vertical)]
            rights = K.zeros((peps.columns + 1,) + endpoint_shape, dtype="complex128")
            rights = K.scatter(
                rights, K.convert_to_tensor([[peps.columns]]), endpoint[None]
            )

            def right_step(carry, x):
                env, buffer = carry
                if vertical:
                    env = right2(env, lower[x], top[x], grid[y, x], grid[y + 1, x])
                else:
                    env = right1(env, lower[x], top[x], grid[y, x])
                buffer = K.scatter(buffer, K.reshape(x, (1, 1)), env[None])
                return env, buffer

            _, rights = K.scan(
                right_step, K.arange(peps.columns - 1, -1, -1), (endpoint, rights)
            )
            uniform = uniforms[int(vertical), y]

            def propose(carry, x):
                values, nodes, left, string, accepted = carry
                yn, xn = (y + 1, x) if vertical else (y, x + 1)
                changed = values[y, x] != values[yn, xn]
                candidate = K.scatter(
                    values,
                    K.stack(
                        [
                            K.stack([K.cast(y, "int64"), x]),
                            K.stack([K.cast(yn, "int64"), xn]),
                        ]
                    ),
                    K.stack([values[yn, xn], values[y, x]]),
                )
                a, b = selected(candidate, y, x), selected(candidate, yn, xn)
                if vertical:
                    current = left2(
                        left, lower[x], top[x], nodes[y, x], nodes[y + 1, x]
                    )
                    proposed = left2(string, lower[x], top[x], a, b)
                    ratio = K.sum(proposed * rights[x + 1]) / K.sum(
                        current * rights[x + 1]
                    )
                else:
                    current = left1(
                        left1(left, lower[x], top[x], nodes[y, x]),
                        lower[x + 1],
                        top[x + 1],
                        nodes[y, x + 1],
                    )
                    proposed = left1(
                        left1(left, lower[x], top[x], a), lower[x + 1], top[x + 1], b
                    )
                    ratio = K.sum(proposed * rights[x + 2]) / K.sum(
                        current * rights[x + 2]
                    )
                accept = changed & (uniform[x] < K.abs(ratio) ** 2)
                values = K.where(accept, candidate, values)
                if vertical:
                    row_mask = (K.arange(peps.rows) == y) | (
                        K.arange(peps.rows) == y + 1
                    )
                    mask = (
                        row_mask[:, None]
                        & (K.arange(peps.columns)[None, :] < x)
                        & accept
                    )
                    nodes = nodes * K.where(mask[:, :, None, None, None, None], swap, 1)
                    old = left
                    left, string = K.where(accept, string, left), K.where(
                        accept, old, string
                    )
                indices = K.stack(
                    [
                        K.stack([K.cast(y, "int64"), x]),
                        K.stack([K.cast(yn, "int64"), xn]),
                    ]
                )
                updates = K.stack(
                    [K.where(accept, a, nodes[y, x]), K.where(accept, b, nodes[yn, xn])]
                )
                nodes = K.scatter(nodes, indices, updates)
                if vertical:
                    left = left2(left, lower[x], top[x], nodes[y, x], nodes[y + 1, x])
                    string = left2(
                        string,
                        lower[x],
                        top[x],
                        nodes[y, x] * swap,
                        nodes[y + 1, x] * swap,
                    )
                else:
                    left = left1(left, lower[x], top[x], nodes[y, x])
                return values, nodes, left, string, accepted + K.cast(accept, "float64")

            state, grid, _, _, accepted = K.scan(
                propose,
                K.arange(peps.columns if vertical else peps.columns - 1),
                (state, grid, endpoint, endpoint, K.cast(0.0, "float64")),
            )
            total = total + accepted
            lower, _, _ = apply_grid_row_dmrg(lower, grid[y], chi, peps.contraction_key)
    return K.reshape(state, (-1,)), total / len(peps.bonds)
