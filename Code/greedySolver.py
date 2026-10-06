import itertools
import time

import networkx as nx
import numpy as np

from buildConflictGraph import build_antenna_subgraph, antenna_range


def greedy_mwis(G, weights):
    """
    Classical WEIGHTED_MIS oracle: heaviest first, then fewest conflicts, then
    closest to the antenna. No capacity yet.
    """
    chosen, blocked = [], set()
    for k in sorted(G.nodes, key=lambda k: (-weights[k], G.degree(k), G.nodes[k]["dist"], k)):
        if k in blocked:
            continue
        chosen.append(k)
        blocked |= set(G.neighbors(k))
    return chosen


## Pruning policies: the user with the largest score is dropped first
PRUNE_POLICIES = {
    "furthest":      lambda G, w, k: G.nodes[k]["dist"],
    "lowest_weight": lambda G, w, k: -w[k],
}


def _capacity(capacity, a):
    return int(capacity[a]) if np.ndim(capacity) else int(capacity)


## Serve the users left out, if possible, moving as few as possible.
##Keeping a user on its antenna costs 0, any other link, 1000 + dist in m.
def repair_coverage(scenario, M, capacity, range_m, compatible=None, constraints=()):
    """
    Returns (M_repaired {uid: antenna}, feasible: every user can be served,
             applied: False if discarded because of a constraint).
    """
    D = scenario["distances"]
    L, K = D.shape
    uids = [int(u) for u in scenario["uids"]]

    F = nx.DiGraph()
    for k in range(K):
        F.add_edge("src", ("u", k), capacity=1, weight=0)
        for a in range(L):
            if D[a, k] <= antenna_range(range_m, a) and (compatible is None or compatible(k, a, scenario)):
                cost = 0 if M.get(uids[k]) == a else 1000 + int(round(D[a, k]))
                F.add_edge(("u", k), ("a", a), capacity=1, weight=cost)
    for a in range(L):
        F.add_edge(("a", a), "snk", capacity=_capacity(capacity, a), weight=0)

    flow = nx.max_flow_min_cost(F, "src", "snk")
    R = {uids[k]: node[1] for k in range(K) if ("u", k) in flow
         for node, f in flow[("u", k)].items() if f}
    feasible = len(R) == K

    if constraints:
        seat = {u: k for k, u in enumerate(uids)}
        by_antenna = {}
        for u, a in R.items():
            by_antenna.setdefault(a, []).append(seat[u])
        for a, ks in by_antenna.items():
            for u, v in itertools.combinations(ks, 2):
                if any(check(u, v, a, scenario) for _, _, check in constraints):
                    return dict(M), feasible, False
    return R, feasible, True


def assign_users_to_antennas(scenario, capacity=4, range_m=200.0, M_prev=None, beta=1.0, prune="furthest",
                             constraints=(), compatible=None, base_weight=None, oracle=greedy_mwis,
                             repair=True):
    """
    One timeslot of AssignUsersToAntennas.

    Returns (M_s {uid: antenna}, unserved [uid], info).
     Info:
      antennas:  {antenna: (|V_a|, |MIS|, |S|)}
      status:    "all served" | "repaired" (repair served everyone)
                | "max coverage" (serving everyone is impossible in this slot)
                | "repair discarded" (it broke a constraint) | "unserved" (repair off)
      moved, added   
      users moved to another antenna / newly served by the repair
    """
    uids = [int(u) for u in scenario["uids"]]
    K, L = len(uids), scenario["distances"].shape[0]
    score = PRUNE_POLICIES[prune] if isinstance(prune, str) else prune

    ## 1. Setup
    present = set(uids)
    M_prev = {u: a for u, a in (M_prev or {}).items() if u in present}   ## drop users who left
    n_prior = {a: 0 for a in range(L)}
    for a in M_prev.values():
        n_prior[a] += 1
    n_range = (scenario["distances"] <= np.array([antenna_range(range_m, a) for a in range(L)])[:, None]).sum(axis=1)
    order = sorted(range(L), key=lambda a: (-n_prior[a], -n_range[a], a))   ## most prior users first
    unassigned, M_s, per_antenna = set(range(K)), {}, {}

    ## 2. Antenna loop
    for a in order:
        if not unassigned:
            break

        ## ub graph of a
        G = build_antenna_subgraph(a, sorted(unassigned), scenario, range_m, constraints, compatible)
        if G.number_of_nodes() == 0:
            continue

        ## Weights, boost for users served by a in the previous slot.
        w = {k: 1.0 if base_weight is None else float(base_weight(k, a, scenario)) for k in G.nodes}
        prior = {k for k in G.nodes if M_prev.get(uids[k]) == a}
        if beta > 1:
            for k in prior:
                w[k] *= beta

        ## Maximum weighted independent set
        S = list(oracle(G, w))
        n_mis = len(S)

        ## Capacity: drop by policy. Ties drop users not in Prior_a first, then the furthest
        while len(S) > _capacity(capacity, a):
            S.remove(max(S, key=lambda k: (score(G, w, k), k not in prior, G.nodes[k]["dist"], k)))


        for k in S:
            M_s[uids[k]] = a
        unassigned -= set(S)
        per_antenna[a] = (G.number_of_nodes(), n_mis, len(S))

    ## 3. Repair.
    info = {"antennas": per_antenna, "status": "all served", "moved": 0, "added": 0}
    if unassigned:
        if repair:
            R, feasible, applied = repair_coverage(scenario, M_s, capacity, range_m, compatible, constraints)
            info["moved"] = sum(1 for u, a in R.items() if u in M_s and M_s[u] != a)
            info["added"] = sum(1 for u in R if u not in M_s)
            M_s = R
            if not applied:
                info["status"] = "repair discarded"   ## it broke a constraint edge
            else:
                info["status"] = "repaired" if feasible else "max coverage"
        else:
            info["status"] = "unserved"


    unserved = [u for u in uids if u not in M_s]
    return M_s, unserved, info


def schedule_assignment(snapshots, capacity=4, range_m=200.0, beta=1.0, prune="furthest", constraints=(),
                        compatible=None, base_weight=None, oracle=greedy_mwis, repair=True):
    """
    Run AssignUsersToAntennas over consecutive timeslots.

    Returns dict:
      mapping    {t: {uid: antenna}}
      unserved   {t: [uid, ...]}
      info       {t: info of assign_users_to_antennas (per antenna sizes, status, moved, added)}
      status     {t: status}: "all served", "repaired", "max coverage" or "unserved"
      snapshots, parameters, runtime_s
    """
    start = time.perf_counter()
    mapping, unserved, info = {}, {}, {}
    M_prev = None
    for t, s in enumerate(snapshots):
        mapping[t], unserved[t], info[t] = assign_users_to_antennas(
            s, capacity, range_m, M_prev=M_prev, beta=beta, prune=prune, constraints=constraints,
            compatible=compatible, base_weight=base_weight, oracle=oracle, repair=repair,
        )
        M_prev = mapping[t]

    return {
        "mapping": mapping,
        "unserved": unserved,
        "info": info,
        "status": {t: i["status"] for t, i in info.items()},
        "snapshots": snapshots,
        "T": len(snapshots),
        "capacity": capacity,
        "range_m": range_m,
        "beta": beta,
        "prune": prune if isinstance(prune, str) else getattr(prune, "__name__", "custom"),
        "repair": repair,
        "constraints": constraints,
        "compatible": compatible,
        "runtime_s": time.perf_counter() - start,
    }


if __name__ == "__main__":

    from dynamic_scenario import DynamicScenario, CARRIER_HZ
    from buildConflictGraph import nearest_antennas
    from evaluate import evaluate_assignment, print_report_assignment, print_assignment

    ## Scenario
    K          = 60
    L          = 30
    T          = 10
    SEED       = 2
    CHANGE     = (0.2, 0.5)
    CHURN      = 0.5
    MAX_MOVE_M = 90.0          ## pedestrians in ~1 min

    ## Antennas
    CAPACITY   = 4  
    RANGE_M    = 200.0

    ## Solver
    BETA       = 1.0           ## > 1 favours users the antenna served in the previous slot
    PRUNE      = "furthest"    ## or "lowest_weight"
    NEAREST    = None          ## int: an antenna only takes users for which it is one of their nearest
    REPAIR     = True          ## serve every user when possible
    MAX_SE     = 5.55          ## bits/s/Hz cap (uplink 64QAM); None = Shannon

    snapshots = DynamicScenario(
        L=L, K=K, semilla=SEED, change_range=CHANGE, churn_share=CHURN, max_move_m=MAX_MOVE_M,
        channel_Hz=CARRIER_HZ / CAPACITY,
    ).run(T)
    result = schedule_assignment(
        snapshots, CAPACITY, RANGE_M, beta=BETA, prune=PRUNE, repair=REPAIR,
        compatible=nearest_antennas(NEAREST) if NEAREST else None,
    )
    report = evaluate_assignment(result, max_se=MAX_SE)

    print(f"K={K} users, L={L} antennas, capacity {CAPACITY} ({CARRIER_HZ / CAPACITY / 1e6:g} MHz per user), "
          f"range {RANGE_M:g} m, T={T}, seed {SEED}")
    print_report_assignment(result, report)

    print("  Assignment per timeslot (antenna: users; + new user, ~ changed antenna)")
    for t in range(T):
        print_assignment(result, t)
