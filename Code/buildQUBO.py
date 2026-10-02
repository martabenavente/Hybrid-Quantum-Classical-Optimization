import itertools
import numpy as np
import networkx as nx


def build_primary_graph(G):
    """Conflict graph edges + same UE ("primary") edges, on the same link nodes."""

    if G.graph.get("capacity", 1) != 1:
        raise NotImplementedError("Option C is formulated for N = 1 (antenna copies needed for N > 1).")
    H = nx.Graph(capacity=1)
    H.add_nodes_from(G.nodes(data=True))
    for a, b, d in G.edges(data=True):
        H.add_edge(a, b, kinds=set(d["kinds"]))
    links_by_ue = {}
    for n, d in G.nodes(data=True):
        links_by_ue.setdefault(d["ue"], []).append(n)
    for links in links_by_ue.values():
        for a, b in itertools.combinations(sorted(links), 2):
            if H.has_edge(a, b):
                H.edges[a, b]["kinds"].add("primary")
            else:
                H.add_edge(a, b, kinds={"primary"})
    return H


def build_qubo(G, weights=None, bonus_margin=1.0, alpha_margin=1.0):
    """
    Build the QUBO of one slot.

    G: conflict graph from build_conflict_graph (N = 1)
    weights  {link: weight} for this slot (default: link SE)
    Returns a dict with:
      variables: list of links, qubit i <-> variables[i]
      Q: (n x n) upper triangular matrix, E(x) = x^T Q x
      H: primary link graph
      w_norm, w_prime, W, alpha: the coefficients, for inspection
    """
    H = build_primary_graph(G)
    variables = sorted(H.nodes)
    index = {n: i for i, n in enumerate(variables)}
    n_ues = len({d["ue"] for _, d in H.nodes(data=True)})

    if weights is None:
        weights = {n: H.nodes[n]["weight"] for n in variables}
    w = np.array([weights[n] for n in variables], dtype=float)
    w_norm = w / w.max() if w.max() > 0 else w

    top = np.sort(w_norm)[::-1][: max(n_ues - 1, 0)]
    W = float(top.sum()) + bonus_margin
    w_prime = w_norm + W

    Q = np.zeros((len(variables), len(variables)))
    Q[np.diag_indices_from(Q)] = -w_prime
    alpha = {}
    for a, b in H.edges:
        i, j = sorted((index[a], index[b]))
        alpha[(a, b)] = max(w_prime[i], w_prime[j]) + alpha_margin
        Q[i, j] = alpha[(a, b)]

    return {
        "variables": variables,
        "index": index,
        "Q": Q,
        "H": H,
        "w_norm": dict(zip(variables, w_norm)),
        "w_prime": dict(zip(variables, w_prime)),
        "W": W,
        "alpha": alpha,
        "bonus_margin": bonus_margin,
        "alpha_margin": alpha_margin,
    }


def energy(qubo, x):
    """E(x) = x^T Q x for a 0/1 vector (or a matrix of them, one per row)."""
    x = np.asarray(x, dtype=float)
    Q = qubo["Q"]
    if x.ndim == 1:
        return float(x @ Q @ x)
    return np.einsum("si,ij,sj->s", x, Q, x)


def encode(qubo, links):
    x = np.zeros(len(qubo["variables"]))
    for n in links:
        x[qubo["index"][n]] = 1
    return x


def decode(qubo, x):
    return [n for n, xi in zip(qubo["variables"], x) if xi > 0.5]


def qubo_to_dict(qubo):
    """{(i, j): coefficient}, the usual input format for QUBO samplers."""
    Q = qubo["Q"]
    return {(i, j): float(Q[i, j]) for i, j in zip(*np.nonzero(Q))}


def complete_slot(G, primary_links, weights=None):
    """
    Classical step after the sampler: keep the primary links and give every AP
    they leave idle to its best remaining link.
    """
    if weights is None:
        weights = {n: G.nodes[n]["weight"] for n in G.nodes}
    chosen = set(primary_links)
    blocked = set().union(*(set(G.neighbors(n)) for n in chosen)) if chosen else set()
    for n in sorted(G.nodes, key=lambda n: (-weights[n], n)):
        if n not in chosen and n not in blocked:
            chosen.add(n)
            blocked |= set(G.neighbors(n))
    return sorted(chosen)


def summary(qubo):
    H, Q = qubo["H"], qubo["Q"]
    kinds = {}
    for *_, d in H.edges(data=True):
        for k in d["kinds"]:
            kinds[k] = kinds.get(k, 0) + 1
    n = len(qubo["variables"])
    coeffs = np.abs(Q[np.nonzero(Q)])
    w = np.array(list(qubo["w_norm"].values()))
    return {
        "qubits": n,
        "quadratic_terms": H.number_of_edges(),
        "density": H.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0.0,
        "edges_by_kind": kinds,
        "W": qubo["W"],
        "coeff_range": (float(coeffs.min()), float(coeffs.max())),
        "smallest_weight_gap": float(np.min(np.diff(np.unique(w)))) if len(np.unique(w)) > 1 else 0.0,
    }


if __name__ == "__main__":

    from generate_scenario import get_data
    from buildConflictGraph import build_conflict_graph
    from greedySolver import pf_schedule

    L, N = 30, 1

    for K, A, cluster in [(10, 5, [1, 2, 8]), (15, 3, [3, 11, 12, 14])]:
        active_APs, _, gainOverNoisedB, powgain, _, pilotIndex, interference_matrix = get_data(
            L=L, K=K, N=N, tau_p=4, ASD_varphi=10 * (3.14159 / 180), numActiveAPs=A, grid=True, semilla=2
        )
        G = build_conflict_graph(active_APs, powgain, gainOverNoisedB, pilotIndex,
                                 interference_matrix, N=N)

        ## Slot 0 of the PF scheduler: PF weights = SE / eps
        qubo = build_qubo(G)
        s = summary(qubo)
        print("=" * 84)
        print(f"K={K} UEs x {A} active APs  (L={L}, N={N}, seed=2)  --  QUBO of slot 0")
        print(f"  Qubits: {s['qubits']}   quadratic terms: {s['quadratic_terms']}   density: {s['density']:.1%}")
        print(f"  Edges by kind: {s['edges_by_kind']}   (an edge can carry more than one kind)")
        print(f"  Coverage bonus W = {s['W']:.3f}   |coefficients| in [{s['coeff_range'][0]:.3f}, {s['coeff_range'][1]:.3f}]"
              f"   smallest weight difference: {s['smallest_weight_gap']:.2e}")

        ## Energy of the greedy's slot, from the matrix vs from the formula
        greedy_slot = pf_schedule(G, 1)["schedule"][0]
        primary = []
        for k in sorted({n[0] for n in greedy_slot}):
            mine = [n for n in greedy_slot if n[0] == k]
            primary.append(max(mine, key=lambda n: (G.nodes[n]["weight"], -n[1])))
        e_matrix = energy(qubo, encode(qubo, primary))
        e_formula = -sum(qubo["w_prime"][n] for n in primary)
        independent = not any(qubo["H"].has_edge(a, b) for a, b in itertools.combinations(primary, 2))
        print("  Check 1 - greedy slot 0, one primary link per UE:")
        print(f"    independent in H: {independent}   E(matrix) = {e_matrix:.6f}   "
              f"E(formula) = {e_formula:.6f}   match: {abs(e_matrix - e_formula) < 1e-9}")
        e_full = energy(qubo, encode(qubo, greedy_slot))
        print(f"    full greedy slot ({len(greedy_slot)} links, several per UE): E = {e_full:.3f} "
              f"(> primary-only, as expected: same-UE pairs are penalised)")
        print(f"    complete_slot(primary) reproduces the greedy slot: "
              f"{complete_slot(G, primary) == greedy_slot}")

        ## Exhaustive check on a real contested cluster
        sub = G.subgraph([n for n in G.nodes if n[0] in cluster]).copy()
        q = build_qubo(sub)
        n = len(q["variables"])
        states = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        E = energy(q, states)
        order = np.argsort(E, kind="stable")
        best = decode(q, states[order[0]])

        H = q["H"]
        def is_independent(links):
            return not any(H.has_edge(a, b) for a, b in itertools.combinations(links, 2))
        def covers(links):
            return {m[0] for m in links} == set(cluster)

        ## Reference, computed separately: best "one AP per UE, all APs different"
        options = [[m for m in q["variables"] if m[0] == k] for k in cluster]
        ref, ref_w = None, -np.inf
        for combo in itertools.product(*options):
            if len({m[1] for m in combo}) == len(combo) and is_independent(combo):
                wsum = sum(q["w_norm"][m] for m in combo)
                if wsum > ref_w:
                    ref, ref_w = sorted(combo), wsum

        invalid = [i for i in order if not (is_independent(decode(q, states[i])) and covers(decode(q, states[i])))]
        print(f"  Check 2 - exhaustive, UEs {cluster} ({n} qubits, {2**n} states):")
        print(f"    lowest-energy state: {best}")
        print(f"    independent: {is_independent(best)}   covers every UE: {covers(best)}   "
              f"= best covering assignment: {best == ref}")
        print(f"    energy gap to the best INVALID state: {E[invalid[0]] - E[order[0]]:.3f}")
        print()
