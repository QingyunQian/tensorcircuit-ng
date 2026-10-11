"""Distribute 6 x 6 PEPS chains across the visible JAX devices.

Contraction batches trade parallelism against temporary memory. Full-sample
batches expose chain parallelism when their compiler memory estimate fits.
The physics and random stream are shared with run_peps_6x6.py.
"""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

if __package__:
    from .run_peps_6x6 import main
else:
    from run_peps_6x6 import main


def distribute(theta, states, key):
    """Place independent chains on devices and replicate the shared PEPS and RNG."""
    devices = jax.devices()
    if len(devices) < 2 or states.shape[0] % len(devices):
        raise ValueError("Use at least two visible devices and a divisible chain count")
    mesh = Mesh(np.asarray(devices), ("chains",))
    replicated = NamedSharding(mesh, PartitionSpec())
    chains = NamedSharding(mesh, PartitionSpec("chains", None))
    return (
        jax.device_put(theta, replicated),
        jax.device_put(states, chains),
        jax.device_put(key, replicated),
    )


if __name__ == "__main__":
    main(placement=distribute)
