import time
import numpy as np
import networkx as nx

from buildConflictGraph import good_links


def _ues(G):
    return sorted({d["ue"] for _, d in G.nodes(data=True)})


def can_cover(ues, links, cap, blocked=frozenset()):
    """
    Matching check: can every UE get its own AP from links?
    """
    ues = list(ues)
    if not ues:
        return True
    ue_nodes = [("ue", k) for k in ues]
    ue_set = set(ues)
    B = nx.Graph()
    B.add_nodes_from(ue_nodes)
    for (k, l) in links:
        if k in ue_set and (k, l) not in blocked:
            for c in range(cap.get(l, 0)):
                B.add_edge(("ue", k), ("ap", l, c))
    matching = nx.bipartite.hopcroft_karp_matching(B, top_nodes=ue_nodes)
    return all(u in matching for u in ue_nodes)


def greedy_covering_mwis(G, weights, N):
    """
    Classical oracle for one slot: a high weight independent set of links that
    covers every UE and respects AP capacity N.
    """
    links = list(G.nodes)
    cap = {G.nodes[n]["ap"]: N for n in links}
    uncovered = set(_ues(G))

    if not can_cover(uncovered, links, cap):
        return None, "No assignment covers every UE (more UEs than reachable AP capacity)"

    chosen, blocked = [], set()
    for link in sorted(links, key=lambda n: (-weights[n], n)):
        k, l = link
        if link in blocked or cap[l] == 0:
            continue
        nbrs = set(G.neighbors(link))
        cap[l] -= 1
        if can_cover(uncovered - {k}, links, cap, blocked | nbrs):
            chosen.append(link)
            blocked |= nbrs
            uncovered.discard(k)
        else:
            cap[l] += 1

    if uncovered:
        return None, f"UEs {sorted(uncovered)} could not be covered"
    return sorted(chosen), None



class PFScheduler:
    """PF scheduler: call step(G) once per slot."""

    def __init__(self, beta=0.5, eps=1e-6, good_margin_db=3.0, oracle=greedy_covering_mwis):
        self.beta = beta
        self.eps = eps
        self.good_margin_db = good_margin_db
        self.oracle = oracle
        self.avg_rate = {}          ## UE: exponential moving average of its rate

    def pf_weights(self, G):
        for k in _ues(G):
            self.avg_rate.setdefault(k, self.eps)   ## new UE: top priority
        pf = {n: d["weight"] / self.avg_rate[d["ue"]] for n, d in G.nodes(data=True)}
        good = good_links(G, self.good_margin_db)
        if len(good) == len(pf):
            return pf
        ## No good links: keep their order by SE, but scale them strictly below the
        ## weakest good link, so they can only take an AP that no good link wants.
        floor = min(pf[n] for n in good)
        max_se = max(G.nodes[n]["weight"] for n in pf if n not in good) or 1.0
        for n in pf:
            if n not in good:
                pf[n] = 0.5 * floor * G.nodes[n]["weight"] / max_se
        return pf

    def step(self, G):
        N = G.graph.get("capacity", 1)
        weights = self.pf_weights(G)
        chosen, why = self.oracle(G, weights, N)
        if chosen is None:
            return None, why
        rate = {k: 0.0 for k in _ues(G)}
        for n in chosen:
            rate[n[0]] += G.nodes[n]["weight"]
        for k, r in rate.items():
            self.avg_rate[k] = (1 - self.beta) * self.avg_rate[k] + self.beta * r
        return chosen, None


def pf_schedule(G, T, beta=0.5, eps=1e-6, good_margin_db=3.0, oracle=greedy_covering_mwis):
    """
    Schedule T consecutive slots on G with the PF scheduler.

    Returns dict:
      schedule         {t: [(ue, ap), ...]}   links active in slot t
      status, reason
      avg_rate         {t: {ue: avg rate after t}}
      runtime_s
    """
    start = time.perf_counter()
    sched = PFScheduler(beta=beta, eps=eps, good_margin_db=good_margin_db, oracle=oracle)
    schedule, avg_rate = {}, {}
    status, reason = "ok", None

    for t in range(T):
        chosen, why = sched.step(G)
        if chosen is None:
            status, reason = "infeasible", f"slot {t}: {why}"
            break
        schedule[t] = chosen
        avg_rate[t] = dict(sched.avg_rate)

    return {
        "schedule": schedule,
        "T": T,
        "N": G.graph.get("capacity", 1),
        "beta": beta,
        "good_margin_db": good_margin_db,
        "status": status,
        "reason": reason,
        "avg_rate": avg_rate,
        "runtime_s": time.perf_counter() - start,
    }


if __name__ == "__main__":

    from generate_scenario import get_data
    from buildConflictGraph import build_conflict_graph
    from evaluate import evaluate, print_report

    configs = [
        dict(K=10, numActiveAPs=5),   ## 50 links
        dict(K=15, numActiveAPs=3),   ## 45 links
    ]
    L, N, BETA, MARGIN = 30, 1, 0.5, 3.0
    T_DETAIL, T_LONG = 3, 10

    variants = [
        ("no fairness",           dict(beta=0.0,  good_margin_db=np.inf)),
        ("PF, all links",         dict(beta=BETA, good_margin_db=np.inf)),
        (f"PF, good links {MARGIN:g}dB", dict(beta=BETA, good_margin_db=MARGIN)),
    ]

    for cfg in configs:
        K, A = cfg["K"], cfg["numActiveAPs"]
        active_APs, _, gainOverNoisedB, powgain, _, pilotIndex, interference_matrix = get_data(
            L=L, K=K, N=N, tau_p=4, ASD_varphi=10 * (3.14159 / 180), numActiveAPs=A, grid=True, semilla=2
        )
        G = build_conflict_graph(active_APs, powgain, gainOverNoisedB, pilotIndex,
                                 interference_matrix, N=N)

        ## Full schedule
        label, params = variants[-1]
        result = pf_schedule(G, T_DETAIL, **params)
        print("=" * 84)
        print(f"K={K} UEs x {A} active APs  (L={L}, T={T_DETAIL}, N={N}, seed=2)  --  {label}")
        print_report(G, result, evaluate(G, result, good_margin_db=MARGIN))

        ## Comparison of all variants
        print("-" * 84)
        print(f"K={K} x {A}: summary")
        print(f"  {'':24s} {'T':>3s} {'SE/slot':>8s} {'min UE':>8s} {'good cov':>9s} {'worst good':>11s} "
              f"{'handovers/slot':>15s}  valid")
        for T in (T_DETAIL, T_LONG):
            for label, params in variants:
                r = evaluate(G, pf_schedule(G, T, **params), good_margin_db=MARGIN)
                print(f"  {label:24s} {T:3d} {r['total_weight']/T:8.3f} {r['min_ue_rate']:8.3f} "
                      f"{r['good_coverage']:9.0%} {r['min_good_share']:11.0%} "
                      f"{r['handovers']/max(T-1, 1):15.2f}  {r['valid']}")
        print()
