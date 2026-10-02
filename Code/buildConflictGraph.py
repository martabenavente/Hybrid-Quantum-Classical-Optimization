import itertools
import networkx as nx
import numpy as np


def link_sinr(powgain, interference_matrix, k, l):
    """SINR of the link AP l -> UE k (linear)."""
    return float(powgain[l, k] / (1 + interference_matrix[l, k]))


def link_weight(powgain, interference_matrix, k, l):
    """Spectral efficiency of the link AP l -> UE k: log2(1 + SINR)."""
    return float(np.log2(1 + link_sinr(powgain, interference_matrix, k, l)))


def build_conflict_graph(
    active_APs,
    powgain,
    gainOverNoisedB,
    pilotIndex,
    interference_matrix,
    N=1,                  ## antennas per AP
    tau_db=3.0,           ## gain margin (dB) for a pilot contamination edge
    pilot_edges=False,    ## pilot contamination family: off by default, redundant when N=1
):
    """
    Returns G: graph whose nodes are (UE, AP) links and whose edges connect
    links that cannot share a timeslot.
    """

    K = active_APs.shape[0]

    G = nx.Graph(capacity=N)

    ## Build nodes: one per (UE, AP) link in the UE's active cluster
    links_by_ap = {}
    for k in range(K):
        for l in active_APs[k]:
            l = int(l)
            node = (k, l)
            G.add_node(
                node,
                ue=k,
                ap=l,
                weight=link_weight(powgain, interference_matrix, k, l),
                sinr_db=float(10 * np.log10(link_sinr(powgain, interference_matrix, k, l))),
                pilot=int(pilotIndex[k]),
            )
            links_by_ap.setdefault(l, []).append(node)

    def add_conflict(n1, n2, kind):
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kinds"].add(kind)
        else:
            G.add_edge(n1, n2, kinds={kind})

    ## Same AP: AP capacity and pilot contamination both live on links that share an AP
    for l, links in links_by_ap.items():
        for n1, n2 in itertools.combinations(links, 2):
            k1, k2 = n1[0], n2[0]

            ## AP capacity (pairwise only valid for N = 1)
            if N == 1:
                add_conflict(n1, n2, "ap_capacity")

            ## Pilot contamination: same pilot at the same AP, comparable gain.
            if pilot_edges and pilotIndex[k1] == pilotIndex[k2]:
                interference_db = gainOverNoisedB[l, k2] - gainOverNoisedB[l, k1]
                if interference_db > -tau_db:
                    add_conflict(n1, n2, "pilot")

    return G


def good_links(G, margin_db=3.0):
    """
    Good"¡ links: SINR within margin_db of their UE's best link.
    """
    best = {}
    for _, d in G.nodes(data=True):
        best[d["ue"]] = max(best.get(d["ue"], -np.inf), d["sinr_db"])
    return {n for n, d in G.nodes(data=True) if d["sinr_db"] >= best[d["ue"]] - margin_db}


def ap_load(G):
    """Number of UEs competing for each AP."""
    load = {}
    for _, data in G.nodes(data=True):
        load[data["ap"]] = load.get(data["ap"], 0) + 1
    return load


if __name__ == "__main__":

    from generate_scenario import get_data

    configs = [
        dict(K=10, numActiveAPs=5),   ## 50 links
        dict(K=15, numActiveAPs=3),   ## 45 links
    ]
    L, T = 30, 3   ## more APs than UEs in both configs

    for cfg in configs:
        K, A = cfg["K"], cfg["numActiveAPs"]

        active_APs, df_simulation, gainOverNoisedB, powgain, max_gain, pilotIndex, interference_matrix = get_data(
            L=L, K=K, N=1, tau_p=4, ASD_varphi=10 * (3.14159 / 180), numActiveAPs=A, grid=True, semilla=2
        )

        G = build_conflict_graph(
            active_APs, powgain, gainOverNoisedB, pilotIndex, interference_matrix,
            N=1, tau_db=3.0,
        )

        load = ap_load(G)
        shared = {l: n for l, n in load.items() if n > 1}

        print("=" * 70)
        print(f"K={K} UEs x {A} active APs  (L={L}, T={T})")
        print(f"  Expected nodes: {K*A}   Actual: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        n_pilot = sum("pilot" in d["kinds"] for _, _, d in G.edges(data=True))
        print(f"  Edges also tagged as pilot contamination: {n_pilot}")
        print(f"  APs used: {len(load)} of {L}   shared APs: {len(shared)}   "
              f"max UEs on one AP: {max(load.values())}")

        weights = [d["weight"] for _, d in G.nodes(data=True)]
        print(f"  Link weights (bits/s/Hz): min={min(weights):.3f}  "
              f"mean={np.mean(weights):.3f}  max={max(weights):.3f}")

        print("  Links per UE (AP: weight):")
        for k in range(K):
            row = ", ".join(f"{l}: {G.nodes[(k, int(l))]['weight']:.2f}" for l in active_APs[k])
            print(f"    UE {k:2d} (pilot {pilotIndex[k]}): {row}")
