import itertools
import json
import os

import networkx as nx
import numpy as np


def slack_coefficients(C):
    """Bounded binary encoding of 0..C: 1, 2, 4, ..., remainder."""
    coeffs, total, p = [], 0, 1
    while total + p <= C:
        coeffs.append(p)
        total += p
        p *= 2
    if total < C:
        coeffs.append(C - total)
    return coeffs


def build_antenna_qubo(G, weights, capacity=None, penalty_factor=1.5):
    """
    QUBO of antenna graph G with node weights {u: w_u}.
    
    Returns dict: Q (symmetric), offset, users (variable i <-> users[i] for
    i < n_users, then slack bits), n_users, slack, P, lam.
    """
    users = sorted(G.nodes)
    n = len(users)
    idx = {u: i for i, u in enumerate(users)}
    w_max = max(weights.values()) if weights else 1.0
    P = lam = penalty_factor * w_max

    use_capacity = capacity is not None and n > capacity
    slack = slack_coefficients(int(capacity)) if use_capacity else []
    N = n + len(slack)
    Q = np.zeros((N, N))
    offset = 0.0

    for u in users:
        Q[idx[u], idx[u]] -= weights[u]
    for u, v, d in G.edges(data=True):
        c = P if d["hard"] else d["weight"]
        Q[idx[u], idx[v]] += c / 2
        Q[idx[v], idx[u]] += c / 2

    if use_capacity:
        a = np.array([1.0] * n + [float(c) for c in slack])
        C = float(capacity)
        Q += lam * np.outer(a, a) - np.diag(2 * lam * C * a)
        offset += lam * C ** 2

    return {"Q": Q, "offset": offset, "users": users, "n_users": n, "slack": slack, "P": P, "lam": lam}


def energy(qubo, x):
    """E(x) = x^T Q x + offset, for one 0/1 vector or one per row."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(x @ qubo["Q"] @ x) + qubo["offset"]
    return np.einsum("si,ij,sj->s", x, qubo["Q"], x) + qubo["offset"]


def decode(qubo, x):
    """Users with x_u = 1 (slack bits dropped)."""
    return [u for u, xi in zip(qubo["users"], x[: qubo["n_users"]]) if xi > 0.5]


def antenna_objective(G, weights, S):
    """sum w - sum soft J over the users S; -inf if S contains a hard pair."""
    value = sum(weights[u] for u in S)
    for u, v in itertools.combinations(S, 2):
        if G.has_edge(u, v):
            if G.edges[u, v]["hard"]:
                return -np.inf
            value -= G.edges[u, v]["weight"]
    return value


def exact_antenna(G, weights, capacity):
    """Exact optimum by enumerating every set of <= capacity users. Returns (value, users)."""
    best, best_S = 0.0, []
    for r in range(1, min(capacity, G.number_of_nodes()) + 1):
        for S in itertools.combinations(sorted(G.nodes), r):
            v = antenna_objective(G, weights, S)
            if v > best + 1e-12:
                best, best_S = v, list(S)
    return best, best_S


def qubo_summary(qubo):
    """Size and coefficients of a QUBO: variables, couplings, density, coefficient range."""
    Q, N = qubo["Q"], qubo["Q"].shape[0]
    off = Q[np.triu_indices(N, 1)]
    coupled = off[off != 0]
    coeffs = np.abs(np.concatenate([np.diag(Q), off]))
    coeffs = coeffs[coeffs > 0]
    return {
        "variables": N,
        "users": qubo["n_users"],
        "slack": len(qubo["slack"]),
        "couplings": int(coupled.size),
        "density": coupled.size / (N * (N - 1) / 2) if N > 1 else 0.0,
        "off_diagonal_nonnegative": bool((off >= 0).all()),
        "coeff_min": float(coeffs.min()) if coeffs.size else 0.0,
        "coeff_max": float(coeffs.max()) if coeffs.size else 0.0,
    }


def qubo_formula(qubo):
    """E(x) written out with the numbers of the QUBO (x_i users, s_j slack bits)."""
    Q, N, n = qubo["Q"], qubo["Q"].shape[0], qubo["n_users"]
    names = [f"x{i}" for i in range(n)] + [f"s{j}" for j in range(N - n)]
    terms = [f"{Q[i, i]:+.3f}*{names[i]}" for i in range(N) if Q[i, i] != 0]
    terms += [f"{2 * Q[i, j]:+.3f}*{names[i]}*{names[j]}"
              for i in range(N) for j in range(i + 1, N) if Q[i, j] != 0]
    if qubo["offset"]:
        terms.append(f"{qubo['offset']:+.3f}")
    return "E(x) = " + " ".join(terms) if terms else "E(x) = 0"


def print_qubo(G, weights, qubo, title=""):
    """The QUBO of one antenna step by step: users and weights, edges, formula, matrix."""
    n, users = qubo["n_users"], qubo["users"]
    N = qubo["Q"].shape[0]
    idx = {u: i for i, u in enumerate(users)}
    print(f"  {title}")
    print("    Variables: x_i = 1 if the antenna serves user i"
          + (f"; slack bits s_j with coefficients {qubo['slack']}" if qubo["slack"] else ""))
    print(f"    {'var':>4s} {'uid':>5s} {'dist m':>7s} {'SE':>6s} {'w_u':>7s}")
    for i, u in enumerate(users):
        d = G.nodes[u]
        print(f"    {'x' + str(i):>4s} {d['uid']:5d} {d['dist']:7.0f} {d['se']:6.2f} {weights[u]:7.3f}")
    cap_txt = f", capacity penalty lambda = {qubo['lam']:.3f}" if qubo["slack"] else ", capacity not in the QUBO"
    print(f"    Edges ({G.number_of_edges()}): hard penalty P = {qubo['P']:.3f}{cap_txt}")
    for u, v, d in sorted(G.edges(data=True), key=lambda e: (idx[e[0]], idx[e[1]])):
        cost = f"hard (P = {qubo['P']:.3f})" if d["hard"] else f"soft J = {d['weight']:.3f}"
        print(f"      x{idx[u]}-x{idx[v]}: {cost:22s} {sorted(d['kinds'])}")
    print(f"    {qubo_formula(qubo)}")
    names = [f"x{i}" for i in range(n)] + [f"s{j}" for j in range(N - n)]
    print(f"    Q (symmetric, E = x^T Q x + offset, offset = {qubo['offset']:.3f}):")
    print("    " + " " * 5 + "".join(f"{c:>8s}" for c in names))
    for i in range(N):
        row = "".join(f"{qubo['Q'][i, j]:8.3f}" if qubo["Q"][i, j] else f"{'.':>8s}" for j in range(N))
        print(f"    {names[i]:>5s}{row}")

def problem_record(t, G, weights, capacity, qubo, reference=None):
    """JSON-ready record of one antenna problem: graph, weights, capacity, QUBO, references."""
    users = qubo["users"]
    return {
        "slot": t,
        "antenna": int(G.graph["antenna"]),
        "capacity": int(capacity),
        "users": [int(u) for u in users],
        "uids": [int(G.nodes[u]["uid"]) for u in users],
        "dist": [G.nodes[u]["dist"] for u in users],
        "se": [G.nodes[u]["se"] for u in users],
        "weights": [float(weights[u]) for u in users],
        "edges": [[int(u), int(v), float(d["weight"]), bool(d["hard"]), sorted(d["kinds"])]
                  for u, v, d in G.edges(data=True)],
        "Q": qubo["Q"].tolist(),
        "offset": qubo["offset"],
        "n_users": qubo["n_users"],
        "slack": qubo["slack"],
        "P": qubo["P"],
        "lam": qubo["lam"],
        "summary": qubo_summary(qubo),
        "reference": reference or {},
    }


def save_qubos(path, params, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "problems": records}, f)


def load_qubos(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def problem_graph(rec):
    """(G, weights, capacity) of a saved problem, as the oracles get them."""
    G = nx.Graph(antenna=rec["antenna"])
    for u, uid, d, se in zip(rec["users"], rec["uids"], rec["dist"], rec["se"]):
        G.add_node(u, uid=uid, dist=d, se=se)
    for u, v, wt, hard, kinds in rec["edges"]:
        G.add_edge(u, v, weight=wt, hard=hard, kinds=set(kinds))
    return G, dict(zip(rec["users"], rec["weights"])), rec["capacity"]


def problem_qubo(rec):
    """The saved QUBO, in the form of build_antenna_qubo (for energy / decode)."""
    return {"Q": np.array(rec["Q"]), "offset": rec["offset"], "users": rec["users"],
            "n_users": rec["n_users"], "slack": rec["slack"], "P": rec["P"], "lam": rec["lam"]}


## Hub split: decide the users with many couplings before the QPU
##   x_h = 0: h and its edges are removed.
##   x_h = 1: h is removed and counted as served (value + w_h, capacity - 1),
##            its hard partners are removed (they can no longer be served),
##            each soft neighbour v pays the coupling on its own: w_v -> w_v - J_hv
##            (DMM). Users left with w <= 0
##            are removed too (with couplings >= 0 they can never pay off).
## The best branch is the optimum of the original problem.

def condition_branch(G, weights, capacity, h, served):
    """Branch of (G, weights, capacity) with user h fixed. Returns (G, weights, capacity, value of h)."""
    if not served:
        keep = [u for u in G.nodes if u != h]
        return G.subgraph(keep).copy(), {u: weights[u] for u in keep}, capacity, 0.0
    w = {u: weights[u] for u in G.nodes if u != h}
    for v in G.neighbors(h):
        if G.edges[h, v]["hard"]:
            w.pop(v, None)
        else:
            w[v] -= G.edges[h, v]["weight"]
    keep = [u for u in w if w[u] > 1e-12] if capacity > 1 else []
    return G.subgraph(keep).copy(), {u: w[u] for u in keep}, capacity - 1, float(weights[h])


def split_hubs(G, weights, capacity, max_degree=6, max_hubs=2):
    """
    Branches with the hubs decided.
    Returns a list of dicts: G, weights, capacity, served (hubs fixed to 1), value (sum of their weights).
    Without hubs: one branch, the problem itself.
    """
    branches = [{"G": G, "weights": dict(weights), "capacity": capacity, "served": [], "value": 0.0, "fixed": 0}]
    for _ in range(max_hubs):
        new, split = [], False
        for b in branches:
            deg = dict(b["G"].degree())
            hub = max(deg, key=lambda u: (deg[u], b["weights"][u], -u), default=None)
            if hub is None or deg[hub] <= max_degree or b["capacity"] < 1:
                new.append(b)
                continue
            split = True
            for served in (False, True):
                Gb, wb, cb, val = condition_branch(b["G"], b["weights"], b["capacity"], hub, served)
                new.append({"G": Gb, "weights": wb, "capacity": cb, "served": b["served"] + ([hub] if served else []),
                            "value": b["value"] + val, "fixed": b["fixed"] + 1})
        branches = new
        if not split:
            break
    return branches


def exact_split(G, weights, capacity, max_degree=6, max_hubs=2):
    """Optimum via the hub branches: must equal exact_antenna."""
    best, best_S = 0.0, []
    for b in split_hubs(G, weights, capacity, max_degree, max_hubs):
        v, S = exact_antenna(b["G"], b["weights"], b["capacity"]) if b["G"].number_of_nodes() else (0.0, [])
        if b["value"] + v > best + 1e-12:
            best, best_S = b["value"] + v, b["served"] + S
    return best, best_S


if __name__ == "__main__":

    from dynamic_scenario import DynamicScenario, CARRIER_HZ
    from buildConflictGraph import min_separation, leakage, best_antennas, adjacent_probability
    from greedySolver import greedy_oracle, assign_users_to_antennas

    ## Same setup as the pipeline
    K, L, T, SEED = 120, 30, 10, 2
    CAPACITY, RANGE_M, BEST_APS, MAX_QUBO_USERS = 6, 300.0, 3, 25
    ALPHA_QUALITY, ALPHA_CONTINUITY, MAX_SE = 1.0, 0.5, 5.55
    MIN_SEP_M, ALPHA_LEAK, LEAKAGE_DB = 25.0, 2.0, 30.0
    P_ADJACENT = adjacent_probability(CAPACITY)   ## two users of an antenna on adjacent channels
    CAPACITY_IN_QUBO = False          ## False: capacity left to the prune step
    PENALTY_FACTOR = 1.5
    EXAMPLE = None                    ## (slot, antenna) to show in full, None: the largest QUBO
    OUT = os.path.join("results", f"qubos_K{K}_C{CAPACITY}_seed{SEED}.json")

    ## 1. The per-antenna problems of a run (each oracle call: graph, weights, capacity)
    constraints = [min_separation(MIN_SEP_M), leakage(ALPHA_LEAK, LEAKAGE_DB, MAX_SE, p_adjacent=P_ADJACENT)]
    snapshots = DynamicScenario(L=L, K=K, semilla=SEED, channel_Hz=CARRIER_HZ / CAPACITY).run(T)
    problems, current = [], {}

    def collect(G, w, cap):
        problems.append((current["t"], G.copy(), dict(w), cap))
        return greedy_oracle(G, w, cap)

    M_prev = None
    for t, s in enumerate(snapshots):
        current["t"] = t
        M_prev, _, _ = assign_users_to_antennas(
            s, CAPACITY, RANGE_M, M_prev=M_prev, alpha_quality=ALPHA_QUALITY, alpha_continuity=ALPHA_CONTINUITY,
            max_se=MAX_SE, constraints=constraints, compatible=best_antennas(BEST_APS),
            oracle=collect, max_candidates=MAX_QUBO_USERS)

    ## 2. Their QUBOs
    qubos = [build_antenna_qubo(G, w, cap if CAPACITY_IN_QUBO else None, PENALTY_FACTOR)
             for _, G, w, cap in problems]
    print(f"K={K}, L={L}, T={T}, seed {SEED}, capacity {CAPACITY} "
          f"({'in the QUBO' if CAPACITY_IN_QUBO else 'by prune, not in the QUBO'}): {len(qubos)} antenna QUBOs")
    print("  E(x) = - sum w_u x_u + sum_soft J_uv x_u x_v + P sum_hard x_u x_v"
          + (" + lam (sum x_u + sum c_j s_j - C)^2" if CAPACITY_IN_QUBO else ""))
    print(f"  w_u = {ALPHA_QUALITY:g} * SE(u,a) / {MAX_SE:g} + {ALPHA_CONTINUITY:g} * [a served u in s-1]   "
          f"J_uv = {ALPHA_LEAK:g} * expected SE lost to leakage (adjacent with p = {P_ADJACENT:.2f}) / {MAX_SE:g}   "
          f"hard: users < {MIN_SEP_M:g} m apart")

    ## 3. One QUBO in full
    if EXAMPLE is None:
        i_ex = max(range(len(qubos)), key=lambda i: (qubos[i]["Q"].shape[0], problems[i][1].number_of_edges()))
    else:
        i_ex = next(i for i, p in enumerate(problems) if (p[0], p[1].graph["antenna"]) == tuple(EXAMPLE))
    t_ex, G_ex, w_ex, cap_ex = problems[i_ex]
    print()
    print_qubo(G_ex, w_ex, qubos[i_ex], f"Example: slot {t_ex}, antenna {G_ex.graph['antenna']}, capacity {cap_ex}")

    ## 4. All QUBOs
    S = [qubo_summary(q) for q in qubos]
    v = np.array([s["variables"] for s in S])
    over = sum(G.number_of_nodes() > cap for _, G, _, cap in problems)
    print()
    print(f"  All QUBOs: variables mean {v.mean():.1f}, 90% {np.percentile(v, 90):.0f}, max {v.max()}   "
          f"more users than capacity: {over}/{len(qubos)}")
    print(f"    couplings mean {np.mean([s['couplings'] for s in S]):.1f}, density mean "
          f"{np.mean([s['density'] for s in S]):.0%}   |coefficients| in "
          f"[{min(s['coeff_min'] for s in S):.3f}, {max(s['coeff_max'] for s in S):.2f}]   "
          f"off-diagonal >= 0 in all: {all(s['off_diagonal_nonnegative'] for s in S)}")

    ## 5. Check and reference values.
    ##    References saved per QUBO: ground state, exact optimum with capacity.
    records, ok, checked = [], 0, 0
    for (t, G, w, cap), q in zip(problems, qubos):
        ref, N = {}, q["Q"].shape[0]
        if N <= 22:
            X = ((np.arange(2 ** N)[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
            E = energy(q, X)
            if not q["slack"]:
                values = np.array([antenna_objective(G, w, decode(q, x)) for x in X])
                feasible = values > -np.inf
                ok += bool(np.allclose(E[feasible], -values[feasible]) and (E[~feasible] > E.min()).all())
            else:
                ok += abs(-E.min() - exact_antenna(G, w, cap)[0]) < 1e-9
            checked += 1
            ref["ground_energy"] = float(E.min())
            ref["ground_users"] = [int(u) for u in decode(q, X[int(np.argmin(E))])]
        best, best_S = exact_antenna(G, w, cap)
        ref["optimum_value"] = float(best)
        ref["optimum_users"] = [int(u) for u in best_S]
        records.append(problem_record(t, G, w, cap, q, ref))
    print(f"  Formulation check: {ok}/{checked} QUBOs correct on every state (<= 22 variables)")
    if not CAPACITY_IN_QUBO:
        within = sum(len(r["reference"].get("ground_users", [])) <= r["capacity"] for r in records)
        print(f"    ground state already within capacity (no prune needed): {within}/{checked}")

    ## 6. Save
    params = dict(K=K, L=L, T=T, seed=SEED, capacity=CAPACITY, range_m=RANGE_M, best_aps=BEST_APS,
                  max_qubo_users=MAX_QUBO_USERS, alpha_quality=ALPHA_QUALITY, alpha_continuity=ALPHA_CONTINUITY,
                  max_se=MAX_SE, min_sep_m=MIN_SEP_M, alpha_leak=ALPHA_LEAK, leakage_db=LEAKAGE_DB, p_adjacent=P_ADJACENT,
                  capacity_in_qubo=CAPACITY_IN_QUBO, penalty_factor=PENALTY_FACTOR)
    save_qubos(OUT, params, records)
    print(f"  Saved {len(records)} QUBOs to {OUT}")
