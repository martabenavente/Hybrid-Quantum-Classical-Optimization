import itertools

from buildConflictGraph import good_links


# ----------------------------------------------------------------------------------
# Validation and metrics
#
# Requirements:
#   - At most T slots are used
#   - Every UE is served by at least one AP in every slot
#   - No AP serves more than N UEs in a slot
#   - No two links in the same slot are joined by a conflict edge
#   - Every scheduled link is a real (UE, AP) link of the graph
# ----------------------------------------------------------------------------------


def evaluate(G, result, good_margin_db=3.0):
    T, N = result["T"], result["N"]
    schedule = result["schedule"]
    ues = sorted({d["ue"] for _, d in G.nodes(data=True)})
    violations = []

    if result["status"] != "ok":
        violations.append(f"solver status: {result['status']} ({result['reason']})")
    if len(schedule) > T:
        violations.append(f"uses {len(schedule)} slots, more than T={T}")

    weight_per_slot, coverage_per_slot = {}, {}
    for t, links in schedule.items():
        for n in links:
            if n not in G:
                violations.append(f"slot {t}: {n} is not a link of the graph")
        links = [n for n in links if n in G]

        ## AP capacity
        per_ap = {}
        for k, l in links:
            per_ap[l] = per_ap.get(l, 0) + 1
        for l, c in per_ap.items():
            if c > N:
                violations.append(f"slot {t}: AP {l} serves {c} UEs (N={N})")

        ## Conflict edges inside the slot
        for a, b in itertools.combinations(links, 2):
            if G.has_edge(a, b):
                violations.append(f"slot {t}: conflict {a} -- {b} ({sorted(G.edges[a, b]['kinds'])})")

        ## Coverage
        served = {k for k, _ in links}
        missing = [k for k in ues if k not in served]
        if missing:
            violations.append(f"slot {t}: UEs not served {missing}")
        coverage_per_slot[t] = len(served) / len(ues)
        weight_per_slot[t] = sum(G.nodes[n]["weight"] for n in links)

    missing_slots = [t for t in range(T) if t not in schedule]
    if missing_slots:
        violations.append(f"slots without a schedule: {missing_slots}")

    ## Fairness.
    ue_rate = {k: {t: 0.0 for t in schedule} for k in ues}
    for t, links in schedule.items():
        for n in links:
            if n in G:
                ue_rate[n[0]][t] += G.nodes[n]["weight"]
    ue_mean_rate = {k: (sum(r.values()) / T if T else 0.0) for k, r in ue_rate.items()}
    x = list(ue_mean_rate.values())
    jain = (sum(x) ** 2) / (len(x) * sum(v * v for v in x)) if x and any(x) else 0.0

    ## AP handovers: an AP that serves a different UE than in the previous slot
    handovers = 0
    slots = sorted(schedule)
    for t_prev, t in zip(slots, slots[1:]):
        prev = {l: k for k, l in schedule[t_prev]}
        for k, l in schedule[t]:
            if l in prev and prev[l] != k:
                handovers += 1

    ## Good link coverage per UE: share of the T slots with >= 1 good link
    good = good_links(G, good_margin_db)
    ue_good_slots = {k: [] for k in ues}
    for t in slots:
        with_good = {n[0] for n in schedule[t] if n in good}
        for k in ues:
            if k in with_good:
                ue_good_slots[k].append(t)
    ue_good_share = {k: (len(s) / T if T else 0.0) for k, s in ue_good_slots.items()}

    return {
        "valid": not violations,
        "violations": violations,
        "total_weight": sum(weight_per_slot.values()),
        "weight_per_slot": weight_per_slot,
        "coverage_per_slot": coverage_per_slot,
        "coverage_rate": (sum(coverage_per_slot.values()) / T) if T else 0.0,
        "slots_used": len(schedule),
        "links_used": len({n for links in schedule.values() for n in links}),
        "ue_rate": ue_rate,
        "ue_mean_rate": ue_mean_rate,
        "min_ue_rate": min(x) if x else 0.0,
        "jain_index": jain,
        "handovers": handovers,
        "good_margin_db": good_margin_db,
        "ue_good_slots": ue_good_slots,
        "ue_good_share": ue_good_share,
        "good_coverage": (sum(ue_good_share.values()) / len(ues)) if ues else 0.0,
        "min_good_share": min(ue_good_share.values()) if ues else 0.0,
        "ues_never_good": [k for k, s in ue_good_share.items() if s == 0],
        "runtime_s": result["runtime_s"],
    }


def print_report(G, result, report):
    print(f"  Valid: {report['valid']}   status: {result['status']}")
    for v in report["violations"]:
        print(f"    ! {v}")
    print(f"  Total SE: {report['total_weight']:.3f} bits/s/Hz   "
          f"coverage: {report['coverage_rate']:.0%}   slots used: {report['slots_used']}/{result['T']}   "
          f"links used: {report['links_used']}/{G.number_of_nodes()}   "
          f"runtime: {report['runtime_s']*1e3:.1f} ms")
    print(f"  Fairness: min UE mean rate {report['min_ue_rate']:.3f}   "
          f"Jain index {report['jain_index']:.3f}   AP handovers {report['handovers']}")
    print(f"  Good-link coverage ({report['good_margin_db']:g} dB): {report['good_coverage']:.0%} of UE-slots   "
          f"worst UE {report['min_good_share']:.0%}   UEs never on a good link: {report['ues_never_good'] or 'none'}")

    ues = sorted({d["ue"] for _, d in G.nodes(data=True)})
    slots = sorted(result["schedule"])
    good = good_links(G, report["good_margin_db"])
    print("  Schedule: APs serving each UE (UE rate in that slot), * = good link")
    print("    UE  | " + " | ".join(f"slot {t:<17}" for t in slots) + " | mean rate | good slots")
    for k in ues:
        cells = []
        for t in slots:
            aps = [f"{l}*" if (kk, l) in good else f"{l}" for (kk, l) in result["schedule"][t] if kk == k]
            cell = f"{','.join(aps) or '-'} ({report['ue_rate'][k][t]:.2f})"
            cells.append(f"{cell:<22}")
        print(f"    {k:3d} | " + " | ".join(cells)
              + f" | {report['ue_mean_rate'][k]:9.3f} | {report['ue_good_share'][k]:.0%}")
    print("    SE  | " + " | ".join(f"{report['weight_per_slot'][t]:<22.3f}" for t in slots))
