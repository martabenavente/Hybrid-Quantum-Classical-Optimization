import itertools
import networkx as nx
import numpy as np


def build_conflict_graph(
    active_APs,
    powgain,
    gainOverNoisedB,
    pilotIndex,
    interference_matrix,
    T,                    ## number of timeslots
    tau_db=3.0,            ## interference threshold for a pilot contamination edge
):
    """
    Returns G: graph where edges connect incompatible nodes.
    """

    K = active_APs.shape[0]
    active_AP_sets = [set(int(l) for l in active_APs[k]) for k in range(K)]

    G = nx.Graph()

    ## Compute weight (spectral efficiency)
    node_weight = {}
    for k in range(K):
        w = sum(
            np.log2(1 + powgain[l, k] / (1 + interference_matrix[l, k]))
            for l in active_AP_sets[k]
        )
        node_weight[k] = w

    ## Build nodes 
    nodes_by_slot = {t: [] for t in range(T)}
    for k, w in node_weight.items():
        for t in range(T):
            node = (k, t)
            G.add_node(node, weight=w)
            nodes_by_slot[t].append(node)

    ## UE exclusivity: same k, different t
    for k in node_weight:
        for t1, t2 in itertools.combinations(range(T), 2):
            G.add_edge((k, t1), (k, t2))

    ## AP capacity and pilot contamination: 2 UEs, same t, same AP; and same pilto, same AP
    ## (they overlap rn because N=1)
    for t in range(T):
        for n1, n2 in itertools.combinations(nodes_by_slot[t], 2):
            k1, k2 = n1[0], n2[0]
            shared_APs = active_AP_sets[k1] & active_AP_sets[k2]
            if not shared_APs:
                continue


            G.add_edge(n1, n2)


            if pilotIndex[k1] == pilotIndex[k2]:
                for l in shared_APs:
                    interference_db = gainOverNoisedB[l, k2] - gainOverNoisedB[l, k1]
                    if interference_db > -tau_db:
                        G.add_edge(n1, n2)
                        break

    return G


# def greedy_mwis(G):
#     """
#     Algorithm 2 (Appendix A, arXiv:2301.02637): iterative greedy maximum
#     weighted independent set extraction.
#     """

#     H = G.copy()
#     selected = []

#     while H.number_of_nodes() > 0:
#         node = max(H.nodes, key=lambda n: H.nodes[n]["weight"])
#         selected.append(node)
#         neighbors = list(H.neighbors(node))
#         H.remove_node(node)
#         H.remove_nodes_from(neighbors)

#     return selected


if __name__ == "__main__":

    from generate_scenario import get_data

    K, T = 15, 3  ## 15 UEs x 3 slots = 45 nodes

    active_APs, df_simulation, gainOverNoisedB, powgain, max_gain, pilotIndex, interference_matrix = get_data(
        L=30, K=K, N=1, tau_p=4, ASD_varphi=10 * (3.14159 / 180), numActiveAPs=5, grid=True, semilla=2
    )

    G = build_conflict_graph(
        active_APs, powgain, gainOverNoisedB, pilotIndex, interference_matrix,
        T=T, tau_db=3.0,
    )

    print(f"Target size: K={K} x T={T} = {K*T} nodes")
    print(f"Actual graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nNodes (k, t) and weights (bits/s/Hz):")
    for node, data in sorted(G.nodes(data=True)):
        print(f"  {node}: weight={data['weight']:.4f}")

    print("\nEdges (conflicts):")
    for n1, n2 in G.edges():
        print(f"  {n1} -- {n2}")
