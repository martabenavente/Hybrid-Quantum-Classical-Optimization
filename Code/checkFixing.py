import json
import os

import numpy as np

from buildQUBO import load_qubos, problem_qubo

EPS = 1e-8   ## same tolerance as qubosolver.transforms.variable_fixing

## qubo-solver applies Hansen's rule recursively
def hansen_recursive(Q):
    """
    Recursive Hansen fixing, tracking what each fixing does.
    Counts how many extra atoms if we keep them fixed.
    """
    N = Q.shape[0]
    live = list(range(N))
    diag = np.diag(Q).astype(float).copy()
    shift = np.zeros(N)
    fixed, rounds, r = {}, {}, 0
    while live:
        sub = Q[np.ix_(live, live)].copy()
        np.fill_diagonal(sub, 0.0)
        new = {}
        for k, i in enumerate(live):
            if diag[i] + 2 * np.minimum(sub[k], 0).sum() >= -EPS:
                new[i] = 0
            elif diag[i] + 2 * np.maximum(sub[k], 0).sum() <= EPS:
                new[i] = 1
        if not new:
            break
        live = [i for i in live if i not in new]
        for u, val in new.items():
            if val == 1:
                for v in live:
                    diag[v] += 2 * Q[u, v]
                    shift[v] += 2 * Q[u, v]
        fixed.update(new)
        rounds.update({i: r for i in new})
        r += 1
    return fixed, rounds, shift, live


def ground_state(Q):
    """Brute force ground state of a small QUBO without offset. Returns x."""
    N = Q.shape[0]
    if N == 0:
        return np.zeros(0)
    X = ((np.arange(2 ** N)[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    return X[int(np.argmin(np.einsum("si,ij,sj->s", X, Q, X)))]


def lift(x_live, live, fixed, N):
    x = np.zeros(N)
    for i, val in fixed.items():
        x[i] = val
    x[live] = x_live
    return x


def library_fixed(Q):
    """Variables fixed by qubo-solver itself (cross check)."""
    import torch
    from qubosolver import Instance
    from qubosolver.transforms import variable_fixing

    inst = variable_fixing.apply_recursively(Instance(matrix=torch.tensor(Q, dtype=torch.float32)))
    return sum(len(d) for d in inst.fixed_indices), inst.size


def emulate(Qr):
    """
    Reduced QUBO Qr on the emulator, without qubo-solver's pre/postprocessing
    (the pulse alone). Returns (bitstrings X, counts).
    """
    import torch
    from qubosolver import Instance, QuantumSolvingConfig, Solver, SolverConfig

    config = SolverConfig(solving=QuantumSolvingConfig(), preprocessing=False, postprocessing=False)
    solution = Solver(Instance(matrix=torch.tensor(Qr, dtype=torch.float32)), config).solve()
    X = solution.bitstrings.detach().cpu().numpy().astype(float)
    counts = solution.counts.detach().cpu().numpy() if solution.counts.numel() else np.ones(len(X))
    return X, counts


if __name__ == "__main__":

    QUBOS = os.path.join("results", "qubos_K120_C6_seed2.json")       ## from buildQUBO.py
    MAX_BRUTE = 22                   ## only for reduced QUBOs up to this size
    CROSS_CHECK = True               ## compare with qubo-solver's own fixing
    EMULATE = True                   ## QUBOs with shifted atoms on the emulator, with and without the shift

    data = load_qubos(QUBOS)
    rows = []
    for rec in data["problems"]:
        q = problem_qubo(rec)
        Q, N = q["Q"], q["Q"].shape[0]
        fixed, rounds, shift, live = hansen_recursive(Q)
        A = (Q != 0) & ~np.eye(N, dtype=bool)                 ## coupling graph
        live_set = set(live)
        one = [i for i, v in fixed.items() if v == 1]
        kind = {}
        for u in one:
            nb = np.flatnonzero(A[u])
            kind[u] = ("isolated" if nb.size == 0 else
                       "live_neighbours" if live_set.intersection(nb.tolist()) else "neighbours_fixed")
        unlocking = [u for u in one if kind[u] == "live_neighbours"]
        affected = [v for v in live if shift[v] > 0]
        w = -np.diag(Q)
        row = {
            "slot": rec["slot"], "antenna": rec["antenna"], "variables": N, "atoms": len(live),
            "fixed_1": len(one), "fixed_0": sum(v == 0 for v in fixed.values()),
            "fixed_1_isolated": sum(kind[u] == "isolated" for u in one),
            "fixed_1_neighbours_fixed": sum(kind[u] == "neighbours_fixed" for u in one),
            "fixed_1_live_neighbours": len(unlocking),
            "fixed_0_by_shift": sum(1 for i, v in fixed.items() if v == 0 and shift[i] > 0),
            "atoms_shifted": len(affected),
            "live_neighbours_per_unlocking": [int(sum(A[u, v] for v in live)) for u in unlocking],
            "rel_shift": [float(shift[v] / w[v]) for v in affected],
            "shift_flips_sign": sum(shift[v] >= w[v] for v in affected),   ## Q_vv would be >= 0: never reached (fixed to 0)
            "atoms_option_a": len(live) + len(unlocking),
            "rounds": 1 + max(rounds.values(), default=-1),
        }
        if affected and len(live) <= MAX_BRUTE:
            Qr = Q[np.ix_(live, live)]
            x_ok = lift(ground_state(Qr + np.diag(shift[live])), live, fixed, N)
            x_no = lift(ground_state(Qr), live, fixed, N)
            e_ok, e_no = x_ok @ Q @ x_ok, x_no @ Q @ x_no
            row["no_shift_wrong"] = bool(e_no > e_ok + 1e-9)
            row["no_shift_gap_rel"] = float((e_no - e_ok) / abs(e_ok)) if e_ok else 0.0
            row["no_shift_extra_users"] = int(x_no.sum() - x_ok.sum())
        if CROSS_CHECK:
            n_fixed, size = library_fixed(Q)
            row["matches_library"] = (n_fixed == len(fixed) and size == len(live))
        if EMULATE and affected:
            ## Ground found among the samples, and share of shots on it, lifted to the original QUBO
            Qr = Q[np.ix_(live, live)]
            x_ok = lift(ground_state(Qr + np.diag(shift[live])), live, fixed, N)
            e_ground = x_ok @ Q @ x_ok
            for tag, M in [("shift", Qr + np.diag(shift[live])), ("no_shift", Qr)]:
                X, counts = emulate(M)
                E = np.array([lift(x, live, fixed, N) @ Q @ lift(x, live, fixed, N) for x in X])
                hit = E <= e_ground + 1e-6
                row[f"emu_{tag}_found"] = bool(hit.any())
                row[f"emu_{tag}_p_ground"] = float(counts[hit].sum() / counts.sum())
                row[f"emu_{tag}_gap_rel"] = float((E.min() - e_ground) / abs(e_ground)) if e_ground else 0.0
            print(f"  emulated slot {rec['slot']} antenna {rec['antenna']}: {len(live)} atoms, ground found "
                  f"with shift {row['emu_shift_found']}, without {row['emu_no_shift_found']}", flush=True)
        rows.append(row)

    ## Print summary
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    has = lambda k: sum(r[k] > 0 for r in rows)
    print(f"{QUBOS}: {n} QUBOs, {tot('variables')} variables")
    if CROSS_CHECK:
        print(f"  same fixing as qubo-solver: {sum(r['matches_library'] for r in rows)}/{n}")
    print(f"  fixed to 1: {tot('fixed_1')} variables ({tot('fixed_1') / tot('variables'):.0%}), "
          f"in {has('fixed_1')}/{n} QUBOs")
    print(f"    isolated (no coupling): {tot('fixed_1_isolated')}   "
          f"all neighbours fixed too: {tot('fixed_1_neighbours_fixed')}   "
          f"with live neighbours (unlock): {tot('fixed_1_live_neighbours')} "
          f"in {has('fixed_1_live_neighbours')}/{n} QUBOs")
    nb = [k for r in rows for k in r["live_neighbours_per_unlocking"]]
    if nb:
        print(f"    live neighbours per unlocking variable: mean {np.mean(nb):.1f}, max {max(nb)}")
    print(f"  fixed to 0: {tot('fixed_0')} ({tot('fixed_0_by_shift')} because of a fixed-to-1 neighbour's shift)")
    print(f"  left as atoms: {tot('atoms')}   QUBOs fully solved by fixing: {sum(r['atoms'] == 0 for r in rows)}/{n}"
          f"   rounds max {max(r['rounds'] for r in rows)}")
    print(f"  atoms that receive a shift: {tot('atoms_shifted')} ({tot('atoms_shifted') / max(tot('atoms'), 1):.0%} "
          f"of atoms), in {has('atoms_shifted')}/{n} QUBOs")
    rel = [x for r in rows for x in r["rel_shift"]]
    if rel:
        print(f"    shift / own weight: mean {np.mean(rel):.0%}, median {np.median(rel):.0%}, max {max(rel):.0%}")
    checked = [r for r in rows if "no_shift_wrong" in r]
    if checked:
        wrong = [r for r in checked if r["no_shift_wrong"]]
        print(f"  without the shift (neither (a) nor (b)): wrong ground state in {len(wrong)}/{len(checked)} "
              f"QUBOs with shifted atoms" + (f", energy {np.mean([r['no_shift_gap_rel'] for r in wrong]):.1%} above "
                                             f"the optimum on average, {np.mean([r['no_shift_extra_users'] for r in wrong]):+.1f} users"
                                             if wrong else ""))
    a = np.array([r["atoms_option_a"] for r in rows])
    b = np.array([r["atoms"] for r in rows])
    print(f"  option (a): {a.sum() - b.sum()} extra atoms in total, {np.sum(a > b)}/{n} QUBOs grow, "
          f"atoms max {b.max()} -> {a.max()}, QUBOs that need the emulator {np.sum(b > 0)} -> {np.sum(a > 0)}")
    print(f"  option (b): no extra atoms; qubo-solver already does it when the device has a DMM "
          f"(drive shaper reads the reduced diagonal)")
    emu = [r for r in rows if "emu_shift_found" in r]
    if emu:
        print(f"  Emulator, pulse alone (no postprocessing), {len(emu)} QUBOs with shifted atoms:")
        for tag, label in [("shift", "with the shift (b)"), ("no_shift", "without the shift")]:
            print(f"    {label:20s} ground state sampled {np.mean([r[f'emu_{tag}_found'] for r in emu]):6.1%}   "
                  f"share of shots on it mean {np.mean([r[f'emu_{tag}_p_ground'] for r in emu]):6.1%}   "
                  f"best sample above ground mean {np.mean([r[f'emu_{tag}_gap_rel'] for r in emu]):.1%}")

    head, name = os.path.split(QUBOS)
    out = os.path.join(head, name.replace("qubos_", "fixing_check_"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"qubos": QUBOS, "rows": rows}, f, default=int)
    print(f"  saved {out}")
