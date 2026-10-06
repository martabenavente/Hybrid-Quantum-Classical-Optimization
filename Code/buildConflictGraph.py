import itertools

import networkx as nx
import numpy as np


def antenna_range(range_m, a):
    """range(a): one value for all antennas, or one per antenna."""
    return float(range_m[a]) if np.ndim(range_m) else float(range_m)


def nearest_antennas(m):
    """compatible(u, a): a is one of the m antennas nearest to u."""
    def compatible(k, a, scenario):
        return a in np.argsort(scenario["distances"][:, k])[:m]
    compatible.__name__ = f"nearest_{m}"
    return compatible


def build_antenna_subgraph(a, unassigned, scenario, range_m, constraints=(), compatible=None):
    """
    Returns graph G_a for antenna a.
      Nodes are unassigned users with dist(u, a) <= range(a) and compatible(u, a).
      Edges are pairs {u, v} violating at least one constraint, with.
    """
    D, powgain, uids = scenario["distances"], scenario["powgain"], scenario["uids"]
    r = antenna_range(range_m, a)
    V = [k for k in unassigned
         if D[a, k] <= r and (compatible is None or compatible(k, a, scenario))]

    G = nx.Graph(antenna=a, range_m=r)
    for k in V:
        G.add_node(
            k,
            uid=int(uids[k]),
            dist=float(D[a, k]),
            snr_db=float(10 * np.log10(powgain[a, k])),
        )

    for u, v in itertools.combinations(V, 2):
        violated = {name: w_c for name, w_c, check in constraints if check(u, v, a, scenario)}
        if violated:
            G.add_edge(u, v, weight=float(sum(violated.values())), kinds=set(violated))

    return G


if __name__ == "__main__":

    from dynamic_scenario import DynamicScenario

    K, RANGE_M = 60, 200.0
    s = DynamicScenario(K=K).snapshot()
    L = s["distances"].shape[0]
    graphs = {a: build_antenna_subgraph(a, range(K), s, RANGE_M) for a in range(L)}
    sizes = [G.number_of_nodes() for G in graphs.values()]
    print(f"K={K} users, L={L} antennas, range {RANGE_M:g} m: users in range per antenna "
          f"mean {np.mean(sizes):.1f}, max {max(sizes)}")
    G = graphs[int(np.argmax(sizes))]
    print(f"  antenna {G.graph['antenna']}: users {sorted(G.nodes)}   edges {G.number_of_edges()}")
