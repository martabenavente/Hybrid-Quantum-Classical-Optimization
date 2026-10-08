import itertools

import numpy as np


# ----------------------------------------------------------------------------------
# AssignUsersToAntennas: mapping {t: {uid: antenna}}, one antenna per user
#
# Requirements (checked against the snapshot of each timeslot):
#   - Every mapped uid is present in the timeslot
#   - Every user is within range(a) of its antenna and compatible with it
#   - No antenna serves more than capacity(a) users
#   - No two users of the same antenna are joined by a constraint edge
# Coverage is a metric, not a requirement: users out of every free antenna wait.
#
# Rate of a served user: min(log2(1 + SINR), max_se). No interference between
# antennas (frequencies are assumed not to be reused enough to interfere). Within
# an antenna, every other user leaks into a user's channel: with probability
# p_adjacent it is on an adjacent channel (leakage_db below its received power,
# 3GPP UE ACLR: 30 dB), otherwise further away (far_db, 43 dB); the expected
# leakage is used. p_adjacent = 2 / capacity for random channels (None: that);
# 1 = all adjacent (worst case). leakage_db None: no leakage.
# max_se is the top of the modulation and coding table: 5.55 bits/s/Hz = uplink
# 64QAM (LTE/NR CQI 15); 7.41 for 256QAM; None for plain Shannon.
# ----------------------------------------------------------------------------------


MAX_SE_REALISTIC = 5.55
LEAKAGE_DB_REALISTIC = 30.0
FAR_DB_REALISTIC = 43.0


def evaluate_assignment(result, max_se=MAX_SE_REALISTIC, leakage_db=LEAKAGE_DB_REALISTIC, p_adjacent=None,
                        far_db=FAR_DB_REALISTIC):
    from buildConflictGraph import antenna_range

    T, snapshots = result["T"], result["snapshots"]
    capacity, range_m = result["capacity"], result["range_m"]
    constraints, compatible = result["constraints"], result["compatible"]
    violations = []
    coverage_per_slot, se_per_slot, uid_rates, dist_served = {}, {}, {}, []
    soft_penalty, soft_pairs = {}, {}       ## per slot: sum of w_c * strength, pairs with a soft cost
    leak_loss = []                          ## per served user: capped SE lost to leakage
    p_adj = (min(1.0, 2.0 / float(np.mean(capacity))) if p_adjacent is None else p_adjacent)

    for t in range(T):
        s, M = snapshots[t], result["mapping"][t]
        seat = {int(u): k for k, u in enumerate(s["uids"])}
        D, powgain = s["distances"], s["powgain"]

        per_antenna = {}
        for u, a in M.items():
            if u not in seat:
                violations.append(f"slot {t}: uid {u} is not in the timeslot")
                continue
            k = seat[u]
            per_antenna.setdefault(a, []).append(k)
            if D[a, k] > antenna_range(range_m, a):
                violations.append(f"slot {t}: uid {u} at {D[a, k]:.0f} m of antenna {a}, out of range")
            if compatible is not None and not compatible(k, a, s):
                violations.append(f"slot {t}: uid {u} not compatible with antenna {a}")

        for a, ks in per_antenna.items():
            cap = int(capacity[a]) if np.ndim(capacity) else int(capacity)
            if len(ks) > cap:
                violations.append(f"slot {t}: antenna {a} serves {len(ks)} users (capacity {cap})")
            for u, v in itertools.combinations(ks, 2):
                for name, w_c, strength, hard in constraints:
                    st = strength(u, v, a, s)
                    if st > 0 and hard:
                        violations.append(f"slot {t}: antenna {a} serves uids {s['uids'][u]}, {s['uids'][v]} ({name})")
                    elif st > 0:
                        soft_penalty[t] = soft_penalty.get(t, 0.0) + w_c * st
                        soft_pairs[t] = soft_pairs.get(t, 0) + 1

        ## Rates: served users log2(1 + SINR), leakage from the other users of their antenna; unserved 0
        g = 0.0 if leakage_db is None else p_adj * 10 ** (-leakage_db / 10) + (1 - p_adj) * 10 ** (-far_db / 10)
        cap = (lambda x: x) if max_se is None else (lambda x: min(x, max_se))
        rates = {}
        for k, u in enumerate(s["uids"]):
            u = int(u)
            a = M.get(u)
            if a is None:
                rates[u] = 0.0
            else:
                leak = g * sum(powgain[a, v] for v in per_antenna[a] if v != k)
                rates[u] = cap(float(np.log2(1 + powgain[a, k] / (1 + leak))))
                leak_loss.append(cap(float(np.log2(1 + powgain[a, k]))) - rates[u])
                dist_served.append(D[a, k])
            uid_rates.setdefault(u, []).append(rates[u])
        coverage_per_slot[t] = len(M) / len(s["uids"])
        se_per_slot[t] = sum(rates.values())

    ## Stability: users present in two consecutive slots
    handovers, lost, kept, arrivals, arrivals_served = 0, 0, 0, 0, 0
    for t in range(1, T):
        prev, now = result["mapping"][t - 1], result["mapping"][t]
        present_before = {int(u) for u in snapshots[t - 1]["uids"]}
        for u in (int(u) for u in snapshots[t]["uids"]):
            if u not in present_before:
                arrivals += 1
                arrivals_served += u in now
            elif u in prev and u in now:
                kept += now[u] == prev[u]
                handovers += now[u] != prev[u]
            elif u in prev:
                lost += 1

    mean_rate = {u: float(np.mean(r)) for u, r in uid_rates.items()}
    x = list(mean_rate.values())
    jain = (sum(x) ** 2) / (len(x) * sum(v * v for v in x)) if x and any(x) else 0.0
    mhz = snapshots[0].get("channel_Hz", 0.0) / 1e6
    loads = [len([u for u, a in result["mapping"][t].items() if a == b])
             for t in range(T) for b in set(result["mapping"][t].values())]

    return {
        "valid": not violations,
        "violations": violations,
        "total_weight": sum(se_per_slot.values()),
        "se_per_slot": se_per_slot,
        "coverage_per_slot": coverage_per_slot,
        "coverage_rate": float(np.mean(list(coverage_per_slot.values()))),
        "unserved": result["unserved"],
        "n_ues": len(mean_rate),
        "uid_mean_rate": mean_rate,
        "min_ue_rate": min(x) if x else 0.0,
        "p10_ue_rate": float(np.percentile(x, 10)) if x else 0.0,
        "median_ue_rate": float(np.median(x)) if x else 0.0,
        "jain_index": jain,
        "channel_MHz": mhz,
        "max_se": max_se,
        "median_mbps": float(np.median(x)) * mhz if x else 0.0,
        "p10_mbps": float(np.percentile(x, 10)) * mhz if x else 0.0,
        "mean_dist_m": float(np.mean(dist_served)) if dist_served else 0.0,
        "antennas_used": float(np.mean([len(set(result["mapping"][t].values())) for t in range(T)])),
        "mean_load": float(np.mean(loads)) if loads else 0.0,
        "leakage_db": leakage_db,
        "p_adjacent": p_adj,
        "leak_hit_share": float(np.mean([x > 0.01 for x in leak_loss])) if leak_loss else 0.0,
        "leak_loss_mean": float(np.mean(leak_loss)) if leak_loss else 0.0,
        "leak_loss_p90": float(np.percentile(leak_loss, 90)) if leak_loss else 0.0,
        "soft_penalty": sum(soft_penalty.values()) / T,   ## per slot
        "soft_pairs": sum(soft_pairs.values()) / T,       ## per slot
        "handovers": handovers,   ## served in both slots, by a different antenna
        "kept": kept,             ## served in both slots, by the same antenna
        "lost": lost,             ## served before, unserved now
        "arrivals": arrivals,
        "arrivals_served": arrivals_served,
        "status_counts": {st: list(result.get("status", {}).values()).count(st)
                          for st in sorted(set(result.get("status", {}).values()))},
        "repair_moved": sum(i.get("moved", 0) for i in result.get("info", {}).values()),
        "repair_added": sum(i.get("added", 0) for i in result.get("info", {}).values()),
        "runtime_s": result["runtime_s"],
    }


def print_report_assignment(result, report):
    T = result["T"]
    print(f"  Valid: {report['valid']}")
    for v in report["violations"]:
        print(f"    ! {v}")
    changed = [len(s["arrived"]) + len(s["moved"]) for s in result["snapshots"][1:]]
    if changed and any(changed):
        K = len(result["snapshots"][0]["uids"])
        print(f"  Scenario: {report['n_ues']} different users over {T} timeslots, "
              f"{np.mean(changed) / K:.0%} changed per timeslot on average")
    print(f"  SE: {report['total_weight'] / T:.2f} bits/s/Hz per slot   coverage {report['coverage_rate']:.0%}   "
          f"runtime {report['runtime_s'] * 1e3:.1f} ms")
    sizes = [s[1] for i in result["info"].values() for s in i["antennas"].values()]
    if sizes:
        capped = sum(s[0] > s[1] for i in result["info"].values() for s in i["antennas"].values())
        print(f"  Oracle problems: {len(sizes)}, users per problem mean {np.mean(sizes):.1f}, "
              f"90% {np.percentile(sizes, 90):.0f}, max {max(sizes)}"
              + (f"   ({capped} capped at {result['max_candidates']})" if result.get("max_candidates") else ""))
    print(f"  Slots: " + ", ".join(f"{n} {st}" for st, n in report["status_counts"].items())
          + f"   repair: {report['repair_added']} users added, {report['repair_moved']} moved")
    print(f"  User mean rate (bits/s/Hz): min {report['min_ue_rate']:.3f}   worst 10% {report['p10_ue_rate']:.3f}   "
          f"median {report['median_ue_rate']:.3f}   Jain {report['jain_index']:.3f}")
    if report["channel_MHz"]:
        cap = "Shannon" if report["max_se"] is None else f"SE capped at {report['max_se']:g} bits/s/Hz"
        print(f"  Throughput ({report['channel_MHz']:g} MHz per user, {cap}): worst 10% {report['p10_mbps']:.1f} Mbps   "
              f"median {report['median_mbps']:.1f} Mbps")
    print(f"  Antennas: {report['antennas_used']:.1f} used per slot, {report['mean_load']:.2f} users each   "
          f"mean distance to serving antenna {report['mean_dist_m']:.0f} m")
    if report["leakage_db"] is not None:
        print(f"  Leakage ({report['leakage_db']:g} dB adjacent with p = {report['p_adjacent']:.2f}, else 43 dB): "
              f"{report['leak_hit_share']:.0%} of "
              f"served users lose rate, mean loss {report['leak_loss_mean']:.2f}, "
              f"worst 10% lose {report['leak_loss_p90']:.2f} bits/s/Hz")
    if any(not c[3] for c in result["constraints"]):
        print(f"  Soft constraints: {report['soft_pairs']:.1f} pairs per slot share an antenna with a cost, "
              f"total {report['soft_penalty']:.2f} per slot")
    stay = report["kept"] + report["handovers"]
    if stay:
        print(f"  Staying users: kept antenna {report['kept']}, handover {report['handovers']} "
              f"({report['handovers'] / stay:.0%}), lost service {report['lost']}")
    if report["arrivals"]:
        print(f"  New users served in their first timeslot: {report['arrivals_served']}/{report['arrivals']}")
    for t, us in report["unserved"].items():
        if us:
            print(f"    slot {t}: unserved uids {us}")


def print_assignment(result, t):
    """Antenna -> users of timeslot t. Marks: + new user, ~ changed antenna since t-1."""
    M = result["mapping"][t]
    prev = result["mapping"].get(t - 1, {})
    present_before = {int(u) for u in result["snapshots"][t - 1]["uids"]} if t > 0 else set()

    def mark(u):
        if t > 0 and u not in present_before:
            return f"{u}+"
        if u in prev and prev[u] != M[u]:
            return f"{u}~"
        return str(u)

    by_antenna = {}
    for u, a in sorted(M.items()):
        by_antenna.setdefault(a, []).append(mark(u))
    status = result.get("status", {}).get(t, "")
    print(f"  Slot {t}: {len(M)} users on {len(by_antenna)} antennas   ({status})")
    for a in sorted(by_antenna):
        print(f"    antenna {a:2d}: {', '.join(by_antenna[a])}")
    if result["unserved"][t]:
        print(f"    unserved: {', '.join(map(str, result['unserved'][t]))}")
