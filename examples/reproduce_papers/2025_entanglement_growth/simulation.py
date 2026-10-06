"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Disorder samples for Figures 1(d,e) and 3. Random streams and sampling
windows match the completed research calculation; each sample is restartable.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from physics import (
    Entropy,
    Evolution,
    Floquet,
    SectorGates,
    gate,
    hamiltonian,
    sector_indices,
)
import tensorcircuit as tc

PROTOCOLS = {
    "thermal": (np.pi / 2, np.pi),
    "A": (0.0, np.pi),
    "B": (np.pi / 2, 0.0),
    "C": (np.pi, 0.0),
    "D": (np.pi, np.pi / 2),
}


def atomic_npz(path, **data):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **data)
    os.replace(temp, path)


def grids(cfg):
    tau = np.array(cfg["tau"])
    baee = np.unique(
        np.r_[
            np.linspace(0, 100, cfg["baee_linear_points"]),
            np.logspace(2, 4, cfg["baee_log_points"]),
        ]
    )
    return tau, baee


def sample(cfg, group, number, output, kernels=None):
    started = time.time()
    L = cfg["L"]
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    path = Path(output) / f"{group}_{number:04d}.npz"
    if path.exists():
        with np.load(path) as old:
            if str(old["config_sha256"]) != digest:
                raise ValueError(f"Config differs from existing checkpoint: {path}")
        print(f"resume: {path}", flush=True)
        return
    Path(output).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], number]))
    idx = sector_indices(L)
    initial_index = int(rng.integers(len(idx)))
    # Independent disorders: preparation, thermal quench, MBL, AL, Floquet,
    # and Fig. 3 preparation, as specified in SM I.
    fields = {
        name: rng.uniform(-W, W, L)
        for name, W in [
            ("prep", 0.5),
            ("thermal", 0.5),
            ("MBL", 5.0),
            ("AL", 5.0),
            ("Floquet", 5.0),
            ("baee", 0.5),
        ]
    }
    tau, baee_times = grids(cfg)
    ent = Entropy(L)

    def hcee(states):
        if kernels is not None:
            return kernels.hcee(states)
        return np.array([ent.value(s) for s in states.T])

    def baee(states):
        if kernels is not None:
            return kernels.baee(states, cuts)
        return ent.average_many(states, cuts)

    psi0 = tc.backend.scatter(
        tc.backend.zeros((len(idx),)), [[initial_index]], [1.0 + 0j]
    )
    cut_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], number, 937]))
    cuts = ent.cuts(cfg.get("max_cuts"), cut_rng)
    out = dict(
        L=L,
        sample=number,
        seed=cfg["seed"],
        config_sha256=digest,
        initial_index=initial_index,
        initial_tc_bitstring=int(idx[initial_index]),
        tau=tau,
        baee_times=baee_times,
        cut_count=len(cuts),
        cuts=np.array(cuts),
        **{f"fields_{k}": v for k, v in fields.items()},
    )
    print(f"start {group} sample={number} L={L}", flush=True)

    if group == "baee":
        ev = Evolution(hamiltonian(L, fields["baee"]))
        states = tc.backend.stack(list(ev.at(psi0, baee_times)), axis=1)
        del ev
        out["baee_hcee"] = hcee(states)
        out["baee_mean"] = baee(states)
    else:
        preparation = Evolution(hamiltonian(L, fields["prep"]))
        states = tc.backend.stack(list(preparation.at(psi0, tau)), axis=1)
        del preparation
        initial = hcee(states)
        out["initial_hcee"] = initial
        if group == "hamiltonian":
            for name, jz in [
                ("thermal", 0.5),
                ("MBL", 0.5),
                ("AL", 0.0),
                ("free", 0.0),
            ]:
                print(f"sample={number} diagonalize {name}", flush=True)
                ev = Evolution(
                    hamiltonian(L, np.zeros(L) if name == "free" else fields[name], jz)
                )
                if name == "free":
                    late = np.zeros(len(tau))
                    for s in ev.at(states, np.arange(201, 301)):
                        late += hcee(s)
                    out["sat_free"] = late / 100
                else:
                    saturated = next(ev.at(states, [1e12]))
                    out[f"sat_{name}"] = hcee(saturated)
                # Stage checkpoints retain useful results if a later stage fails.
                atomic_npz(path.with_name(path.stem + ".partial.npz"), **out)
                if name != "free":
                    del ev
            print(f"sample={number} diagonalize Floquet", flush=True)
            floquet = Floquet(L, fields["Floquet"], ev)
            del ev
            saturated = next(floquet.at(states, [300_000_000_000]))
            out["sat_Floquet"] = hcee(saturated)
        elif group == "rqc":
            out["initial_baee"] = baee(states)
            mapper = SectorGates(L)
            D, realizations = cfg["depth"], cfg["circuit_realizations"]
            bonds = rng.integers(0, L - 1, (realizations, D))
            out["bonds"] = bonds
            late_n = min(100, D)
            for name, angles in PROTOCOLS.items():
                sat = np.zeros((realizations, len(tau)))
                U = gate(*angles)
                print(f"sample={number} RQC {name}", flush=True)
                if kernels is not None:
                    out[f"rqc_sat_{name}"] = kernels.rqc(states, U, bonds)
                else:
                    for r in range(realizations):
                        state = tc.backend.copy(states)
                        for d, bond in enumerate(bonds[r], start=1):
                            state = mapper.apply(state, U, bond)
                            if d > D - late_n:
                                sat[r] += hcee(state)
                    out[f"rqc_sat_{name}"] = sat.mean(axis=0) / late_n
                atomic_npz(path.with_name(path.stem + ".partial.npz"), **out)
        else:
            raise ValueError(group)
    out["elapsed_seconds"] = time.time() - started
    out["tcng_version"] = tc.__version__
    atomic_npz(path, **out)
    path.with_name(path.stem + ".partial.npz").unlink(missing_ok=True)
    print(f"done {path} seconds={out['elapsed_seconds']:.1f}", flush=True)
