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


def best_antennas(m):
    """compatible(u, a): a is one of the m antennas u receives strongest."""
    def compatible(k, a, scenario):
        return a in np.argsort(-scenario["powgain"][:, k])[:m]
    compatible.__name__ = f"best_{m}"
    return compatible


def min_separation(d_min_m=25.0):
    """Hard (blockage): users closer than d_min_m to each other cannot share an antenna."""
    def strength(u, v, a, scenario):
        return float(abs(scenario["ue_pos"][u] - scenario["ue_pos"][v]) < d_min_m)
    return (f"min_sep_{d_min_m:g}m", 1.0, strength, True)


def capped_se(snr, max_se=5.55):
    return min(float(np.log2(1 + snr)), max_se)


def adjacent_probability(capacity):
    """Two users of an antenna with `capacity` channels assigned at random are adjacent with probability 2 / capacity."""
    return min(1.0, 2.0 / capacity)


def pair_leakage_loss(pu, pv, g, max_se=5.55):
    """SE the two users lose when each leaks a fraction g of its received power into the other's channel."""
    return (capped_se(pu, max_se) - capped_se(pu / (1 + g * pv), max_se)
            + capped_se(pv, max_se) - capped_se(pv / (1 + g * pu), max_se))


def leakage(alpha_leak=1.0, leakage_db=30.0, max_se=5.55, min_strength=0.02, p_adjacent=1.0, far_db=43.0):
    """
    Soft (channel leakage): two users of the same antenna on adjacent
    channels lose SE.
    """
    g_adj, g_far = 10 ** (-leakage_db / 10), 10 ** (-far_db / 10)
    def strength(u, v, a, scenario):
        P = scenario["powgain"]
        pu, pv = P[a, u], P[a, v]
        loss = (p_adjacent * pair_leakage_loss(pu, pv, g_adj, max_se)
                + (1 - p_adjacent) * pair_leakage_loss(pu, pv, g_far, max_se))
        st = loss / max_se
        return st if st >= min_strength else 0.0
    return (f"leakage_{leakage_db:g}dB_p{p_adjacent:.2f}", alpha_leak, strength, False)


def build_antenna_subgraph(a, unassigned, scenario, range_m, constraints=(), compatible=None):
    """
    Returns graph G_a for antenna a.
      Nodes are unassigned users with dist(u, a) <= range(a) and compatible(u, a),
      with their distance, SNR and SE (log2(1 + SNR)) at a.
      Edges are pairs {u, v} with strength > 0 in at least one constraint:
      weight = sum of w_c * strength of the soft ones, hard = any hard one.
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
            se=float(np.log2(1 + powgain[a, k])),
        )

    for u, v in itertools.combinations(V, 2):
        soft, hard, kinds = 0.0, False, set()
        for name, w_c, strength, is_hard in constraints:
            s = strength(u, v, a, scenario)
            if s > 0:
                kinds.add(name)
                if is_hard:
                    hard = True
                else:
                    soft += w_c * s
        if kinds:
            G.add_edge(u, v, weight=soft, hard=hard, kinds=kinds)

    return G


if __name__ == "__main__":

    from dynamic_scenario import DynamicScenario

    K, RANGE_M = 60, 200.0
    constraints = [min_separation(25.0), leakage(1.0, 30.0)]
    s = DynamicScenario(K=K).snapshot()
    L = s["distances"].shape[0]
    graphs = {a: build_antenna_subgraph(a, range(K), s, RANGE_M, constraints) for a in range(L)}
    sizes = [G.number_of_nodes() for G in graphs.values()]
    edges = [G.number_of_edges() for G in graphs.values()]
    hard = sum(d["hard"] for G in graphs.values() for *_, d in G.edges(data=True))
    print(f"K={K} users, L={L} antennas, range {RANGE_M:g} m: users per antenna mean {np.mean(sizes):.1f} "
          f"max {max(sizes)}   edges per antenna mean {np.mean(edges):.1f} ({hard} hard in total)")
    G = graphs[int(np.argmax(edges))]
    print(f"  antenna {G.graph['antenna']}: users {sorted(G.nodes)}")
    for u, v, d in G.edges(data=True):
        cost = "hard" if d["hard"] else f"soft {d['weight']:.2f}"
        print(f"    {u}-{v}: {cost}  {sorted(d['kinds'])}")
