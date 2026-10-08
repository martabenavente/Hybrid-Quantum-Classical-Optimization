import itertools
import time

import networkx as nx
import numpy as np

from buildConflictGraph import build_antenna_subgraph, antenna_range


def greedy_oracle(G, weights, capacity):
    """
    Classical oracle: maximise sum w_u x_u - sum J_uv x_u x_v with at most
    'capacity' users and no hard edge inside.
    """
    chosen, gain = [], dict(weights)
    while len(chosen) < capacity and gain:
        k = max(gain, key=lambda k: (gain[k], -G.nodes[k]["dist"], -k))
        if gain[k] <= 0:
            break
        chosen.append(k)
        del gain[k]
        for v in G.neighbors(k):
            if v in gain:
                if G.edges[k, v]["hard"]:
                    del gain[v]
                else:
                    gain[v] -= G.edges[k, v]["weight"]
    return chosen


def node_weights(G, prior, alpha_quality=1.0, alpha_continuity=0.5, max_se=5.55):
    """w_u = alpha_quality * SE(u, a) / max_se (SE capped) + alpha_continuity * [u served by a in s-1]."""
    return {k: alpha_quality * min(G.nodes[k]["se"], max_se) / max_se + alpha_continuity * (k in prior)
            for k in G.nodes}


def set_value(G, weights, S):
    """sum w - sum soft J over the users S; -inf if S contains a hard pair."""
    value = sum(weights[k] for k in S)
    for u, v in itertools.combinations(S, 2):
        if G.has_edge(u, v):
            if G.edges[u, v]["hard"]:
                return -np.inf
            value -= G.edges[u, v]["weight"]
    return value


## Pruning policies: the user with the largest score is dropped first.
## "objective": the user whose removal leaves the best sum w - sum J (default).
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
    Serve as many users as possible, changing M as little as possible.
    Returns (M_repaired {uid: antenna}, all_served).
    """
    D = scenario["distances"]
    L, K = D.shape
    uids = [int(u) for u in scenario["uids"]]
    seat = {u: k for k, u in enumerate(uids)}
    hard = [c for c in constraints if c[3]]
    forbidden = set()                                   ## (k, a) options removed after a conflict

    while True:
        F = nx.DiGraph()
        for k in range(K):
            F.add_edge("src", ("u", k), capacity=1, weight=0)
            for a in range(L):
                if ((k, a) not in forbidden and D[a, k] <= antenna_range(range_m, a)
                        and (compatible is None or compatible(k, a, scenario))):
                    cost = 0 if M.get(uids[k]) == a else 1000 + int(round(D[a, k]))
                    F.add_edge(("u", k), ("a", a), capacity=1, weight=cost)
        for a in range(L):
            F.add_edge(("a", a), "snk", capacity=_capacity(capacity, a), weight=0)

        flow = nx.max_flow_min_cost(F, "src", "snk")
        R = {uids[k]: node[1] for k in range(K) if ("u", k) in flow
             for node, f in flow[("u", k)].items() if f}

        conflicts = set()
        if hard:
            by_antenna = {}
            for u, a in R.items():
                by_antenna.setdefault(a, []).append(seat[u])
            for a, ks in by_antenna.items():
                for u, v in itertools.combinations(ks, 2):
                    if any(strength(u, v, a, scenario) > 0 for _, _, strength, _ in hard):
                        ## the user the repair put there loses the option (M itself has no conflicts)
                        k = u if M.get(uids[u]) != a else v
                        conflicts.add((k, a))
        if not conflicts:
            return R, len(R) == K
        forbidden |= conflicts


def assign_users_to_antennas(scenario, capacity=4, range_m=200.0, M_prev=None, alpha_quality=1.0,
                             alpha_continuity=0.5, max_se=5.55, prune="objective", constraints=(),
                             compatible=None, oracle=greedy_oracle, repair=True, max_candidates=None):
    """
    One timeslot of AssignUsersToAntennas.
    Returns (M_s {uid: antenna}, unserved [uid], info).
     Info:
      antennas:  {antenna: (|V_a|, |oracle input|, |oracle|, |S|)}
      status:    "all served" | "repaired" (repair served everyone)
                | "max coverage" (serving everyone is impossible in this slot)
                | "unserved" (repair off)
      moved, added   
      users moved to another antenna / newly served by the repair
    """
    uids = [int(u) for u in scenario["uids"]]
    K, L = len(uids), scenario["distances"].shape[0]
    score = None if prune == "objective" else PRUNE_POLICIES[prune] if isinstance(prune, str) else prune

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

        ## Weights: link quality + continuity for users served by a in the previous slot
        prior = {k for k in G.nodes if M_prev.get(uids[k]) == a}
        w = node_weights(G, prior, alpha_quality, alpha_continuity, max_se)
        n_candidates = G.number_of_nodes()
        if max_candidates is not None and n_candidates > max_candidates:
            top = sorted(G.nodes, key=lambda k: (-w[k], G.nodes[k]["dist"], k))[:max_candidates]
            G = G.subgraph(top).copy()
            w = {k: w[k] for k in top}

        ## Oracle: max sum w - sum J, at most capacity users, no hard edge
        S = list(oracle(G, w, _capacity(capacity, a)))
        n_mis = len(S)

        ## Capacity (safety net if the oracle returns too many): drop by policy.
        ## Ties drop users not in Prior_a first, then the furthest
        while len(S) > _capacity(capacity, a):
            if score is None:
                drop = max(S, key=lambda k: (set_value(G, w, [x for x in S if x != k]), k not in prior, k))
            else:
                drop = max(S, key=lambda k: (score(G, w, k), k not in prior, G.nodes[k]["dist"], k))
            S.remove(drop)


        for k in S:
            M_s[uids[k]] = a
        unassigned -= set(S)
        per_antenna[a] = (n_candidates, G.number_of_nodes(), n_mis, len(S))

    ## 3. Repair.
    info = {"antennas": per_antenna, "status": "all served", "moved": 0, "added": 0}
    if unassigned:
        if repair:
            R, all_served = repair_coverage(scenario, M_s, capacity, range_m, compatible, constraints)
            info["moved"] = sum(1 for u, a in R.items() if u in M_s and M_s[u] != a)
            info["added"] = sum(1 for u in R if u not in M_s)
            M_s = R
            info["status"] = "repaired" if all_served else "max coverage"
        else:
            info["status"] = "unserved"


    unserved = [u for u in uids if u not in M_s]
    return M_s, unserved, info


def schedule_assignment(snapshots, capacity=4, range_m=200.0, alpha_quality=1.0, alpha_continuity=0.5,
                        max_se=5.55, prune="objective", constraints=(), compatible=None,
                        oracle=greedy_oracle, repair=True, max_candidates=None):
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
            s, capacity, range_m, M_prev=M_prev, alpha_quality=alpha_quality,
            alpha_continuity=alpha_continuity, max_se=max_se, prune=prune, constraints=constraints,
            compatible=compatible, oracle=oracle, repair=repair, max_candidates=max_candidates,
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
        "alpha_quality": alpha_quality,
        "alpha_continuity": alpha_continuity,
        "max_se": max_se,
        "prune": prune if isinstance(prune, str) else getattr(prune, "__name__", "custom"),
        "repair": repair,
        "max_candidates": max_candidates,
        "constraints": constraints,
        "compatible": compatible,
        "runtime_s": time.perf_counter() - start,
    }


if __name__ == "__main__":

    from dynamic_scenario import DynamicScenario, CARRIER_HZ
    from buildConflictGraph import best_antennas, min_separation, leakage, adjacent_probability
    from evaluate import evaluate_assignment, print_report_assignment, print_assignment

    ## Scenario
    K          = 120
    L          = 30
    T          = 10
    SEED       = 2
    CHANGE     = (0.2, 0.5)
    CHURN      = 0.5
    MAX_MOVE_M = 90.0

    ## Antennas
    CAPACITY   = 6
    RANGE_M    = 300.0
    BEST_APS   = 3

    ## Solver: node weights w_u = ALPHA_QUALITY * SE / MAX_SE + ALPHA_CONTINUITY * [same antenna as s-1]
    ALPHA_QUALITY    = 1.0
    ALPHA_CONTINUITY = 0.5          ## as few handovers as possible
    ORACLE     = "greedy"          
    MAX_QUBO_USERS = 25           
    CAPACITY_IN_QUBO = False        
    PRUNE      = "objective"        ## over capacity: drop the user whose removal helps the objective most
    REPAIR     = True               ## serve every user when possible
    MAX_SE     = 5.55               ## bits/s/Hz cap (uplink 64QAM)

    ## Constraints
    MIN_SEP_M  = 25.0               ## blockage: users closer than this cannot share an antenna
    ALPHA_LEAK = 2.0                ## leakage: weight of the SE a pair loses by sharing an antenna
    LEAKAGE_DB = 30.0               ## leakage into an adjacent channel
    P_ADJACENT = None               ## probability two users of an antenna are on adjacent channels. None = 2 / CAPACITY

    constraints = []
    if MIN_SEP_M:
        constraints.append(min_separation(MIN_SEP_M))
    p_adj = adjacent_probability(CAPACITY) if P_ADJACENT is None else P_ADJACENT
    if ALPHA_LEAK:
        constraints.append(leakage(ALPHA_LEAK, LEAKAGE_DB, MAX_SE, p_adjacent=p_adj))

    snapshots = DynamicScenario(
        L=L, K=K, semilla=SEED, change_range=CHANGE, churn_share=CHURN, max_move_m=MAX_MOVE_M,
        channel_Hz=CARRIER_HZ / CAPACITY,
    ).run(T)
    if ORACLE == "greedy":
        oracle = greedy_oracle
    else:
        from solveQUBO import make_qubo_oracle
        oracle = make_qubo_oracle(ORACLE, capacity_in_qubo=CAPACITY_IN_QUBO)

    result = schedule_assignment(
        snapshots, CAPACITY, RANGE_M, alpha_quality=ALPHA_QUALITY, alpha_continuity=ALPHA_CONTINUITY,
        max_se=MAX_SE, prune=PRUNE, repair=REPAIR, constraints=constraints,
        compatible=best_antennas(BEST_APS) if BEST_APS else None, oracle=oracle,
        max_candidates=MAX_QUBO_USERS,
    )
    report = evaluate_assignment(result, max_se=MAX_SE, leakage_db=LEAKAGE_DB, p_adjacent=p_adj)

    print(f"K={K} users, L={L} antennas, capacity {CAPACITY} ({CARRIER_HZ / CAPACITY / 1e6:g} MHz per user), "
          f"range {RANGE_M:g} m, T={T}, seed {SEED}, oracle {getattr(oracle, '__name__', ORACLE)}")
    print_report_assignment(result, report)

    print("  Assignment per timeslot (antenna: users; + new user, ~ changed antenna)")
    for t in range(T):
        print_assignment(result, t)
