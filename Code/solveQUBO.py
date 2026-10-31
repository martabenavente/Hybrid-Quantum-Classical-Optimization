import itertools
import json
import logging
import os
import time
import warnings

import numpy as np

from buildQUBO import antenna_objective, build_antenna_qubo, decode, energy

logging.disable(logging.INFO)


def brute_force(qubo, max_vars=22):
    """Returns (x, energy)."""
    N = qubo["Q"].shape[0]
    if N > max_vars:
        raise ValueError(f"{N} variables, brute force limited to {max_vars}")
    if N == 0:
        return np.zeros(0), qubo["offset"]
    best_x, best_e = None, np.inf
    for start in range(0, 2 ** N, 2 ** 16):
        ids = np.arange(start, min(start + 2 ** 16, 2 ** N))
        X = ((ids[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
        E = energy(qubo, X)
        i = int(np.argmin(E))
        if E[i] < best_e:
            best_x, best_e = X[i], float(E[i])
    return best_x, best_e


def solve_qubo(qubo, method="tabu_search", **solver_kwargs):
    """
    Lowest-energy state found by 'method' (brute force, tabu search, 
    simulated annealing, quantum + pre/postprocessing or quantum_raw).
    Returns (x, meta): x the 0/1 vector, meta with time_s and, for the quantum
    methods, atoms (0 if qubo-solver solved it without the emulator: trivial or
    fully fixed by preprocessing), shots, distinct bitstrings, p_best (share of
    shots on the returned state).
    """
    N = qubo["Q"].shape[0]
    t0 = time.perf_counter()
    if N == 0:
        return np.zeros(0), {"time_s": 0.0}
    if method == "brute_force":
        x, _ = brute_force(qubo)
        return x, {"time_s": time.perf_counter() - t0}

    import torch
    from qubosolver import ClassicalSolvingConfig, Instance, QuantumSolvingConfig, Solver, SolverConfig

    quantum = method in ("quantum", "quantum_raw")
    if quantum:
        from qubosolver.drive_shaping._device_specs import support_dmm
        raw = method == "quantum_raw"
        solving = QuantumSolvingConfig(**solver_kwargs)
        if not support_dmm(solving.device):
            warnings.warn(f"device {solving.device} has no DMM: every atom gets the same detuning, so the "
                          "QUBO diagonal, including the shifts from variables fixed to 1, does not reach the atoms")
        config = SolverConfig(solving=solving, preprocessing=not raw, postprocessing=not raw)
    else:
        config = SolverConfig(solving=ClassicalSolvingConfig(algorithm=method, **solver_kwargs))
    solver = Solver(Instance(matrix=torch.tensor(qubo["Q"], dtype=torch.float32)), config)
    solution = solver.solve()
    X = solution.bitstrings.detach().cpu().numpy().astype(float)
    i = int(np.argmin(energy(qubo, X)))
    meta = {"time_s": time.perf_counter() - t0}
    if quantum:
        register = getattr(solver._solver, "_cached_register", None)
        counts = solution.counts.detach().cpu().numpy() if solution.counts.numel() else np.zeros(0)
        meta.update({
            "atoms": int(register.n_qubits) if register is not None else 0,
            "shots": int(counts.sum()),
            "distinct": int(len(X)),
            "p_best": float(counts[i] / counts.sum()) if counts.sum() else 1.0,
        })
    return X[i], meta


def reduced_hubs(G, weights, max_degree=6, penalty_factor=1.5):
    """Users with more than max_degree couplings after the variable fixing, and the users left."""
    from checkFixing import hansen_recursive
    q = build_antenna_qubo(G, weights, None, penalty_factor)
    _, _, _, live = hansen_recursive(q["Q"])
    A = q["Q"][np.ix_(live, live)] != 0
    np.fill_diagonal(A, False)
    deg = A.sum(1)
    return [q["users"][live[i]] for i in np.argsort(-deg, kind="stable") if deg[i] > max_degree], \
        {q["users"][i] for i in live}


def defer_hubs(G, weights, max_degree=6, penalty_factor=1.5):
    """G without its hubs. Returns (G, removed)."""
    removed = []
    while True:
        hubs, _ = reduced_hubs(G, weights, max_degree, penalty_factor)
        if not hubs:
            return G, removed
        removed.append(hubs[0])
        G = G.subgraph([u for u in G.nodes if u != hubs[0]]).copy()


def add_back(G, weights, S, users):
    """Deferred users added to S (heaviest first) if no hard conflict with S and positive gain."""
    S = list(S)
    for h in sorted(users, key=lambda k: -weights[k]):
        nb = [v for v in S if G.has_edge(h, v)]
        if any(G.edges[h, v]["hard"] for v in nb):
            continue
        if weights[h] - sum(G.edges[h, v]["weight"] for v in nb) > 0:
            S.append(h)
    return S


def truncate_hubs(G, weights, max_degree=6, penalty_factor=1.5):
    """G with the weakest soft couplings of each hub dropped until no hub is left 
       or only hard couplings remain."""
    G, stuck = G.copy(), set()
    while True:
        hubs, live = reduced_hubs(G, weights, max_degree, penalty_factor)
        hubs = [h for h in hubs if h not in stuck]
        if not hubs:
            return G
        h = hubs[0]
        soft = [v for v in G.neighbors(h) if v in live and not G.edges[h, v]["hard"]]
        if not soft:
            stuck.add(h)
            continue
        G.remove_edge(h, min(soft, key=lambda v: (G.edges[h, v]["weight"], v)))


def make_qubo_oracle(method="tabu_search", capacity_in_qubo=False, penalty_factor=1.5, hub_policy=None,
                     max_degree=6, **solver_kwargs):
    """
    Oracle with the signature of greedy_oracle: oracle(G, weights, capacity) -> users.

    Builds the antenna QUBO, solves it with 'method' and decodes it. If the
    solution breaks a hard edge, the user of the pair with the lowest weight is dropped.
    capacity_in_qubo=False: capacity is left to the prune step of  assign_users_to_antennas,
    which works better for every solver.
    hub_policy: None, "defer" or "truncate" for users with more than max_degree 
    couplings after the variable fixing.

    oracle.calls collects, per call: users, variables, feasible as returned, hubs, time, meta.
    """
    def oracle(G, weights, capacity):
        if G.number_of_nodes() == 0:
            return []
        G_q, deferred = G, []
        if hub_policy == "defer":
            G_q, deferred = defer_hubs(G, weights, max_degree, penalty_factor)
        elif hub_policy == "truncate":
            G_q = truncate_hubs(G, weights, max_degree, penalty_factor)
        S, qubo, feasible, meta = [], None, True, {}
        if G_q.number_of_nodes():
            qubo = build_antenna_qubo(G_q, weights, capacity if capacity_in_qubo else None, penalty_factor)
            x, meta = solve_qubo(qubo, method, **solver_kwargs)
            S = decode(qubo, x)
            feasible = antenna_objective(G, weights, S) > -np.inf
        for u, v in itertools.combinations(sorted(S, key=lambda k: -weights[k]), 2):
            if u in S and v in S and G.has_edge(u, v) and G.edges[u, v]["hard"]:
                S.remove(v)
        S = add_back(G, weights, S, deferred)
        oracle.calls.append({"users": G.number_of_nodes(), "variables": qubo["Q"].shape[0] if qubo else 0,
                             "feasible": feasible, "deferred": len(deferred),
                             "dropped_couplings": G.number_of_edges() - G_q.number_of_edges() if hub_policy == "truncate" else 0,
                             **meta})
        return sorted(S, key=lambda k: -weights[k])
    oracle.calls = []
    oracle.__name__ = f"qubo_{method}" + ("_capacity" if capacity_in_qubo else "") + (f"_{hub_policy}" if hub_policy else "")
    return oracle


def solutions_path(qubos_path, method):
    head, name = os.path.split(qubos_path)
    return os.path.join(head, name.replace("qubos_", f"solutions_{method}_"))


if __name__ == "__main__":

    from buildQUBO import load_qubos, problem_graph, problem_qubo

    QUBOS = os.path.join("results", "qubos_K120_C6_seed2.json")
    METHODS = ["tabu_search", "simulated_annealing", "quantum", "quantum_raw"]
    SOLVER_KWARGS = {"simulated_annealing": {"max_iter": 5000}}
    LIMIT = None                     ## solve only the first LIMIT QUBOs, None = all
    MAX_QUBITS = {"quantum_raw": 12}  ## skip larger QUBOs for a method

    data = load_qubos(QUBOS)
    records = data["problems"][:LIMIT] if LIMIT else data["problems"]
    print(f"{QUBOS}: {len(records)} QUBOs  "
          f"(capacity {'in' if data['params']['capacity_in_qubo'] else 'not in'} the QUBO)")

    for method in METHODS:
        out, t0 = [], time.perf_counter()
        for rec in records:
            if rec["summary"]["variables"] > MAX_QUBITS.get(method, np.inf):
                out.append({"slot": rec["slot"], "antenna": rec["antenna"], "skipped": True,
                            "variables": rec["summary"]["variables"], "couplings": rec["summary"]["couplings"]})
                continue
            G, w, cap = problem_graph(rec)
            q = problem_qubo(rec)
            x, meta = solve_qubo(q, method, **SOLVER_KWARGS.get(method, {}))
            S = decode(q, x)
            e = energy(q, x) if len(x) else q["offset"]
            ground = rec["reference"].get("ground_energy")
            out.append({
                "slot": rec["slot"], "antenna": rec["antenna"],
                "variables": rec["summary"]["variables"], "couplings": rec["summary"]["couplings"],
                "x": [int(v) for v in x], "users": [int(u) for u in S], "energy": float(e),
                "ground_found": None if ground is None else bool(e <= ground + 1e-6),
                "feasible": bool(antenna_objective(G, w, S) > -np.inf),
                "within_capacity": len(S) <= cap,
                **meta,
            })
        total = time.perf_counter() - t0

        skipped = sum(o.get("skipped", False) for o in out)
        pairs = [(o, r) for o, r in zip(out, records) if not o.get("skipped")]
        out_s = [o for o, _ in pairs]
        v = np.array([o["variables"] for o in out_s])
        gf = [o["ground_found"] for o in out_s if o["ground_found"] is not None]
        gaps = [o["energy"] - r["reference"]["ground_energy"] for o, r in pairs
                if r["reference"].get("ground_energy") is not None]
        skip_txt = f"  ({skipped} QUBOs with more than {MAX_QUBITS[method]} qubits skipped)" if skipped else ""
        print(f"\n  {method}{skip_txt}", flush=True)
        print(f"    qubits (variables) mean {v.mean():.1f}, 90% {np.percentile(v, 90):.0f}, max {v.max()}   "
              f"couplings mean {np.mean([o['couplings'] for o in out_s]):.1f}")
        if method.startswith("quantum"):
            atoms = np.array([o.get("atoms", 0) for o in out_s])
            used = atoms > 0
            on_qpu = [o for o in out_s if o.get("atoms", 0)]
            print(f"    on the emulator: {used.sum()}/{len(out_s)} QUBOs (rest trivial or fixed by preprocessing), "
                  f"atoms mean {atoms[used].mean() if used.any() else 0:.1f}, max {atoms.max()}")
            if on_qpu:
                print(f"    shots per QUBO {np.mean([o['shots'] for o in on_qpu]):.0f}   "
                      f"distinct bitstrings mean {np.mean([o['distinct'] for o in on_qpu]):.1f}   "
                      f"share of shots on the returned state mean {np.mean([o['p_best'] for o in on_qpu]):.1%}")
        print(f"    ground state found {np.mean(gf):6.1%}   mean energy above ground {np.mean(gaps):.4f}   "
              f"feasible (no hard pair) {np.mean([o['feasible'] for o in out_s]):6.1%}   "
              f"within capacity {np.mean([o['within_capacity'] for o in out_s]):6.1%}")
        print(f"    time per QUBO mean {np.mean([o['time_s'] for o in out_s]) * 1e3:.1f} ms, total {total:.1f} s")

        path = solutions_path(QUBOS, method)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"qubos": QUBOS, "method": method, "solver_kwargs": SOLVER_KWARGS.get(method, {}),
                       "solutions": out}, f)
        print(f"    saved {path}", flush=True)
