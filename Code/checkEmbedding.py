import itertools
import json
import os
import time

import numpy as np

from buildQUBO import (antenna_objective, build_antenna_qubo, decode, energy, load_qubos, problem_graph,
                       problem_qubo, split_hubs)
from checkFixing import hansen_recursive, lift
from solveQUBO import add_back, defer_hubs



def embed_register(Q, config=None):
    """Positions (n, 2) and interaction matrix (n, n) of qubo-solver's register for Q."""
    import torch
    from qubosolver import Instance
    from qubosolver.solver._embedder import _get_embedder
    from qubosolver.solver.config import QuantumSolvingConfig

    config = config or QuantumSolvingConfig()
    instance = Instance(matrix=torch.tensor(Q, dtype=torch.float32))
    register = _get_embedder(instance, config, backend=None).embed()
    ids = list(register.qubits_ids)
    order = np.argsort([int(i) for i in ids])          ## qubit id i <-> variable i
    coords = np.array([np.asarray(register.qubits[ids[k]], dtype=float) for k in order])
    U = np.asarray(register.interaction_matrix(), dtype=float)[np.ix_(order, order)]
    return coords, U, config


def fit_scale(U, c, mask):
    """s minimising ||s U - c|| over the pairs in mask."""
    u, t = U[mask], c[mask]
    return float(u @ t / (u @ u)) if (u @ u) > 0 else 0.0


def brute_ground(Q):
    n = Q.shape[0]
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    E = np.einsum("si,ij,sj->s", X, Q, X)
    return X[int(np.argmin(E))]


def prune(G, w, S, cap):
    """Capacity prune of assign_users_to_antennas (prune='objective')."""
    S = list(S)
    while len(S) > cap:
        S.remove(max(S, key=lambda k: (antenna_objective(G, w, [x for x in S if x != k]), k)))
    return S


def drop_hard(G, w, S):
    """Same feasibility fix as make_qubo_oracle: drop the lighter user of each hard pair."""
    S = sorted(S, key=lambda k: -w[k])
    for u, v in itertools.combinations(list(S), 2):
        if u in S and v in S and G.has_edge(u, v) and G.edges[u, v]["hard"]:
            S.remove(v)
    return S


def embed_and_solve(G, w, cap):
    """
    The QUBO of (G, w) as the atoms encode it (Q diagonal + fitted s U), solved
    exactly, then hard pairs dropped and pruned to cap. Returns (users, info).
    """
    if G.number_of_nodes() == 0:
        return [], {"atoms": 0, "embedded": False}
    q = build_antenna_qubo(G, w, None)
    Q, n = q["Q"], q["Q"].shape[0]
    off = Q - np.diag(np.diag(Q))
    core = [i for i in range(n) if off[i].any()]
    if len(core) < 2:                                   ## no coupling: nothing to embed, exact
        x, info = brute_ground(Q), {"atoms": n, "embedded": False}
    else:
        _, U_core, _ = embed_register(Q[np.ix_(core, core)])
        U = np.zeros((n, n))
        U[np.ix_(core, core)] = U_core
        iu = np.triu_indices(n, 1)
        c, u = 2 * Q[iu], U[iu]
        s = fit_scale(u, c, np.ones(len(c), dtype=bool))
        su = s * u
        Q_emb = np.diag(np.diag(Q)).astype(float)
        Q_emb[iu] = su / 2
        Q_emb[(iu[1], iu[0])] = su / 2
        x = brute_ground(Q_emb)

        ## The same three checks as check_one
        in_core = np.zeros(n, dtype=bool)
        in_core[core] = True
        edge = c > 0
        non_edge = ~edge & in_core[iu[0]] & in_core[iu[1]]
        users = q["users"]
        hard = np.zeros(len(c), dtype=bool)
        pos = {(i, j): k for k, (i, j) in enumerate(zip(*iu))}
        for a, b, d in G.edges(data=True):
            if d["hard"]:
                hard[pos[tuple(sorted((users.index(a), users.index(b))))]] = True
        from scipy.stats import spearmanr
        n_edges = int(edge.sum())
        k = max(1, int(np.ceil(0.2 * n_edges)))
        info = {
            "atoms": n, "embedded": True,
            "residual": float(np.linalg.norm(su - c) / np.linalg.norm(c)),
            "spurious": float(su[non_edge].max() / np.median(c[edge])) if non_edge.any() else 0.0,
            "rank_corr": float(spearmanr(c[edge], u[edge])[0]) if n_edges >= 3 and np.ptp(c[edge]) > 0 else None,
            "top20_overlap": len(set(np.argsort(-c[edge])[:k]) & set(np.argsort(-u[edge])[:k])) / k,
            "non_edge_above_weakest": float(np.mean(u[non_edge] > u[edge].min())) if non_edge.any() else 0.0,
            "hard_ok": bool((su[hard] > max(w.values())).all()) if hard.any() else None,
            "ground_optimal": bool(energy(q, x) <= energy(q, brute_ground(Q)) + 1e-9),
        }
    S = prune(G, w, drop_hard(G, w, decode(q, x)), cap)
    return S, info


def check_split(rec, max_degree=6, max_hubs=2):
    """Hub split, each branch embedded and solved. The best branch (true objective) is kept."""
    G, w, cap = problem_graph(rec)
    best, rows = -np.inf, []
    for b in split_hubs(G, w, cap, max_degree, max_hubs):
        S, info = embed_and_solve(b["G"], b["weights"], b["capacity"])
        value = b["value"] + antenna_objective(b["G"], b["weights"], S)
        deg = dict(b["G"].degree())
        rows.append({**info, "max_degree": max(deg.values(), default=0)})
        best = max(best, value)
    opt = rec["reference"]["optimum_value"]
    emb = [r for r in rows if r["embedded"]]
    return {"slot": rec["slot"], "antenna": rec["antenna"], "branches": len(rows), "branch_metrics": rows,
            "atoms_max": max(r["atoms"] for r in rows), "atoms_total": sum(r["atoms"] for r in rows),
            "residual_median": float(np.median([r["residual"] for r in emb])) if emb else 0.0,
            "branch_max_degree": max(r["max_degree"] for r in rows),
            "value": float(best), "optimum": float(opt), "gap": float((opt - best) / opt) if opt > 0 else 0.0}


def check_one(rec, reduced=True, hub_policy=None):
    """
    The checks on one QUBO. reduced=True: on the QUBO qubo-solver embeds (after
    its variable fixing, shifts included).
    Solutions are judged on the original QUBO.
    reduced=False: on the full QUBO.
    hub_policy="defer": the QUBO without its hubs (solveQUBO.defer_hubs), as the
    oracle builds it. The hubs are added back before the prune (add_back).
    """
    G, w, cap = problem_graph(rec)
    deferred = []
    if hub_policy == "defer":
        G_q, deferred = defer_hubs(G, w)
    q = problem_qubo(rec) if not deferred else build_antenna_qubo(G_q, w, None)
    Q_full, N = q["Q"], q["Q"].shape[0]
    if reduced:
        fixed, _, shift, live = hansen_recursive(Q_full)
        Q = Q_full[np.ix_(live, live)] + np.diag(shift[live])
    else:
        fixed, live, Q = {}, list(range(N)), Q_full
    n = len(live)
    users = [q["users"][i] for i in live]
    iu = np.triu_indices(n, 1)
    c = 2 * Q[iu]                                   ## coefficient of x_i x_j
    edge = c > 0
    hard = np.zeros(n * (n - 1) // 2, dtype=bool)
    pos = {(i, j): k for k, (i, j) in enumerate(zip(*iu))}
    for u, v, d in G.edges(data=True):
        if d["hard"] and u in users and v in users:
            i, j = sorted((users.index(u), users.index(v)))
            hard[pos[(i, j)]] = True

    ## Users with no coupling are always selected (w_u > 0) and need no
    ## interaction+.
    ## BLaDE cannot fit a register without couplings (it divides by zero), 
    ## so only the coupled users are embedded. Uncoupled ones get U = 0, 
    ## as an exact embedding would give.
    off = Q - np.diag(np.diag(Q))
    core = [i for i in range(n) if off[i].any()]
    if len(core) < 2:
        return None
    t0 = time.perf_counter()
    coords, U_core, config = embed_register(Q[np.ix_(core, core)])
    t_embed = time.perf_counter() - t0
    U = np.zeros((n, n))
    U[np.ix_(core, core)] = U_core
    u = U[iu]
    in_core = np.zeros(n, dtype=bool)
    in_core[core] = True
    core_pair = in_core[iu[0]] & in_core[iu[1]]

    s = fit_scale(u, c, np.ones_like(edge))
    su = s * u
    resid = float(np.linalg.norm(su - c) / np.linalg.norm(c)) if np.linalg.norm(c) else 0.0
    resid_edges = float(np.linalg.norm(su[edge] - c[edge]) / np.linalg.norm(c[edge])) if edge.any() else 0.0
    non_edge = ~edge & core_pair
    spurious = float(su[non_edge].max() / np.median(c[edge])) if edge.any() and non_edge.any() else 0.0

    ## Order of the coefficients: the strongest couplings of Q should stay the strongest in U
    from scipy.stats import spearmanr
    n_edges = int(edge.sum())
    rank_corr = float(spearmanr(c[edge], u[edge])[0]) if n_edges >= 3 and np.ptp(c[edge]) > 0 else None
    k = max(1, int(np.ceil(0.2 * n_edges))) if n_edges else 0
    top_overlap = (len(set(np.argsort(-c[edge])[:k]) & set(np.argsort(-u[edge])[:k])) / k) if k else None
    weakest_edge = u[edge].min() if n_edges else np.inf
    non_edge_above = float(np.mean(u[non_edge] > weakest_edge)) if non_edge.any() and n_edges else 0.0
    w_max = float(-np.diag(Q).min())             ## largest (shifted) weight among the atoms
    hard_ok = bool((su[hard] > w_max).all()) if hard.any() else None

    Q_emb = np.diag(np.diag(Q)).astype(float)
    Q_emb[iu] = su / 2
    Q_emb[(iu[1], iu[0])] = su / 2
    x_emb = lift(brute_ground(Q_emb), live, fixed, N)   ## back to the original QUBO
    S_emb = decode(q, x_emb)
    if deferred:                             
        x_ref = lift(brute_ground(Q), live, fixed, N)
        S_ref, e_ref = decode(q, x_ref), energy(q, x_ref)
    else:
        S_ref, e_ref = rec["reference"].get("ground_users", []), rec["reference"]["ground_energy"]
    same_ground = sorted(S_emb) == sorted(S_ref)
    ## The embedded ground state is judged by its energy in the original QUBO.
    ground_optimal = energy(q, x_emb) <= e_ref + 1e-9
    feasible_emb = antenna_objective(G, w, S_emb) > -np.inf

    ## Final users as the oracle returns them (hard pairs dropped, hubs added back), then the prune
    final = lambda S: prune(G, w, add_back(G, w, drop_hard(G, w, S), deferred), cap)
    opt = rec["reference"]["optimum_value"]
    v_ref = antenna_objective(G, w, final(S_ref))
    v_emb = antenna_objective(G, w, final(S_emb))

    m = len(core)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))[np.triu_indices(m, 1)]
    radial = np.linalg.norm(coords - coords.mean(0), axis=1)
    return {
        "slot": rec["slot"], "antenna": rec["antenna"], "variables": N, "n": n, "atoms": m, "edges": int(edge.sum()),
        "deferred": [int(h) for h in deferred],
        "hard_edges": int(hard.sum()), "max_degree": int((off != 0).sum(1).max()),
        "scale": s, "residual": resid, "residual_edges": resid_edges, "spurious": spurious,
        "rank_corr": rank_corr, "top20_overlap": top_overlap, "non_edge_above_weakest": non_edge_above,
        "hard_ok": hard_ok, "same_ground": bool(same_ground), "ground_optimal": bool(ground_optimal),
        "feasible_emb": bool(feasible_emb),
        "value_ref_pruned": float(v_ref), "value_emb_pruned": float(v_emb), "optimum": float(opt),
        "gap_emb": float((opt - v_emb) / opt) if opt > 0 else 0.0,
        "dist_ratio": float(radial.max() / dist.min()) if n > 1 else 0.0,
        "device_ratio": float(config.max_min_dist_ratio),
        "embed_s": t_embed,
        "loss_emb": float(max(0.0, v_ref - v_emb) / opt) if opt > 0 else 0.0,   ## loss caused by the embedding
        ## for the heatmaps: pair coefficients c = 2 Q_ij and s U_ij (upper triangle), hard pairs, largest weight
        "users": [int(u) for u in users], "c": c.tolist(), "su": su.tolist(), "hard": hard.tolist(), "w_max": w_max,
    }


## Status per metric, tables, heatmaps

OK, CHECK, REVISE = 0, 1, 2
STATUS_TEXT = {OK: "✓ OK", CHECK: "! check", REVISE: "✗ revise"}
STATUS_PLAIN = {OK: "OK", CHECK: "check", REVISE: "REVISE"}
STATUS_COLOR = {OK: "#0ca30c", CHECK: "#fab219", REVISE: "#d03b3b"}
STATUS_INK = {OK: "#ffffff", CHECK: "#0b0b0b", REVISE: "#ffffff"}
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SCREEN_PX = 900 


def _band(good, check, higher_better=False):
    """Status from two thresholds: OK if better than good, check if better than check, else revise."""
    if higher_better:
        return lambda v: OK if v > good else CHECK if v > check else REVISE
    return lambda v: OK if v < good else CHECK if v < check else REVISE


## Key, metric, ideal, good enough, format, status, higher is better, short name
RULES = [
    ("residual", "Error between Q and U  ||sU - c|| / ||c||", "0%", "< 10%  (check < 20%)",
     "{:.0%}", _band(0.10, 0.20), False, "Error"),
    ("spurious", "Strongest fake coupling / typical real one", "0", "< 0.5  (check < 1)",
     "{:.2f}", _band(0.5, 1.0), False, "Fake"),
    ("rank_corr", "Rank correlation of couplings, Q vs U", "1", "> 0.8  (check > 0.5)",
     "{:.2f}", _band(0.8, 0.5, True), True, "Rank"),
    ("top20_overlap", "Strongest 20% of couplings still strongest", "100%", "≥ 80%  (check ≥ 50%)",
     "{:.0%}", _band(0.79, 0.49, True), True, "Top 20%"),
    ("non_edge_above_weakest", "No-conflict pairs above the weakest conflict", "0%", "< 10%  (check < 20%)",
     "{:.0%}", _band(0.10, 0.20), False, "No-conflict"),
    ("hard_ok", "Hard conflicts still forbid the pair", "yes", "yes",
     "bool", lambda v: OK if v else REVISE, True, "Hard"),
    ("ground_optimal", "U-QUBO solution optimal in the original", "yes", "yes  (check: no)",
     "bool", lambda v: OK if v else CHECK, True, "Optimal"),
    ("loss_emb", "Loss caused by the embedding, after prune", "0%", "0%  (check < 1%)",
     "{:.1%}", _band(1e-9, 0.01), False, "Loss"),
    ("dist_margin", "Register size: radial/min distance / device limit", "≤ 1", "≤ 1  (fits the device)",
     "{:.2f}", lambda v: OK if v <= 1 + 1e-9 else REVISE, False, "Size"),
]


OUTCOME = {"hard_ok", "ground_optimal", "loss_emb", "dist_margin"}   ## result (or runnable), not structure


def _plain(text):
    """For the Windows console (cp1252): no >= / <= signs."""
    return text.replace("≥", ">=").replace("≤", "<=")


def _fmt(fmt, v):
    if v is None:
        return "–"
    return ("yes" if v else "no") if fmt == "bool" else fmt.format(v)


def _tint(hex_color, alpha=0.22):
    """hex_color at alpha over the surface (opaque, for table cells)."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])
    s = np.array([int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(x)):02x}" for x in alpha * c + (1 - alpha) * s)


def assess(rows):
    """Adds dist_margin, the status of every metric and the verdict to each row (worst status; the
    structure metrics count at most as check, so a QUBO is revised only if its result changes)."""
    for r in rows:
        r["dist_margin"] = r["dist_ratio"] / r["device_ratio"]
        r["status"] = {k: (rule(r[k]) if r.get(k) is not None else None) for k, _, _, _, _, rule, _, _ in RULES}
        ## Revise only if the result changes; structure metrics alone can only ask for a check
        r["verdict"] = max(s if k in OUTCOME else min(s, CHECK) for k, s in r["status"].items() if s is not None)
    return rows


def summary_table(rows):
    """
    One line per metric: median, worst, QUBOs OK and status (OK if >= 90% of QUBOs OK, check if
    >= 75%, else revise; structure metrics at most check, as in the verdict).
    """
    lines = []
    for k, label, ideal, good, fmt, _, higher, _ in RULES:
        rs = [r for r in rows if r.get(k) is not None]
        if not rs:
            continue
        n_ok = sum(r["status"][k] == OK for r in rs)
        if fmt == "bool":
            yours = f"{sum(bool(r[k]) for r in rs)}/{len(rs)} yes"
        else:
            v = np.array([r[k] for r in rs], dtype=float)
            worst = v.min() if higher else v.max()
            yours = f"median {_fmt(fmt, np.median(v))}, worst {_fmt(fmt, worst)}"
        share = n_ok / len(rs)
        status = OK if share >= 0.9 else CHECK if share >= 0.75 else REVISE
        lines.append({"metric": label, "ideal": ideal, "good": good, "yours": yours, "ok": f"{n_ok}/{len(rs)}",
                      "status": status if k in OUTCOME else min(status, CHECK)})
    return lines


def _table_figure(cells, colours, ink, col_labels, col_widths, title, subtitle, path, row_h=0.32):
    import matplotlib.pyplot as plt
    n = len(cells)
    head = 1.35                                         ## inches for title and two subtitle lines
    fig = plt.figure(figsize=(sum(col_widths) + 0.4, head + row_h * (n + 1)), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.02, 0.98, (row_h * (n + 1)) / (head + row_h * (n + 1)) - 0.02])
    ax.axis("off")
    tab = ax.table(cellText=cells, cellColours=colours, colLabels=col_labels, loc="upper left",
                   colWidths=[w / sum(col_widths) for w in col_widths], cellLoc="left", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.PAD = 0.04
        if i == 0:
            cell.set_text_props(color=INK_2, fontsize=8.5, fontweight="bold")
            cell.set_facecolor(SURFACE)
        else:
            cell.set_text_props(color=ink[i - 1][j], fontsize=8.5,
                                fontweight="bold" if colours[i - 1][j] in STATUS_COLOR.values() else "normal")
    fig.text(0.01, 1 - 0.32 / fig.get_figheight(), title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 1 - 0.68 / fig.get_figheight(), subtitle, fontsize=8.5, color=INK_2, va="top")
    fig.set_dpi(min(100, SCREEN_PX / fig.get_figheight()))   ## on screen: fit the height (the file keeps dpi 200)
    fig.savefig(path, dpi=200, facecolor=SURFACE)        ## left open: shown at the end if SHOW


def plot_summary(lines, n_qubos, n_trivial, verdicts, path):
    cells, colours, ink = [], [], []
    for ln in lines:
        s = ln["status"]
        cells.append([ln["metric"], ln["ideal"], ln["good"], ln["yours"], ln["ok"], STATUS_TEXT[s]])
        colours.append([SURFACE] * 5 + [STATUS_COLOR[s]])
        ink.append([INK, INK_2, INK_2, INK, INK, STATUS_INK[s]])
    v = {s: sum(x == s for x in verdicts) for s in (OK, CHECK, REVISE)}
    _table_figure(cells, colours, ink, ["Metric", "Ideal", "Good enough (rule of thumb)", "Yours", "QUBOs OK", "Status"],
                  [3.6, 0.6, 2.1, 2.3, 0.9, 0.9], "Embedding check: QUBO vs interaction matrix",
                  f"{n_qubos} QUBOs with ≥ 2 couplings ({n_trivial} with a single coupling are embedded exactly and left out).   "
                  f"Per QUBO: {v[OK]} OK, {v[CHECK]} check, {v[REVISE]} revise.\n"
                  f"Status: OK if ≥ 90% of QUBOs are OK, check if ≥ 75%. Revise only for result metrics "
                  f"(hard conflicts, loss, register fits); structure metrics are at most check.", path)


def plot_per_qubo(rows, path):
    rows = sorted(rows, key=lambda r: (-r["verdict"], -r["residual"]))
    cells, colours, ink = [], [], []
    for r in rows:
        line, col, ik = [f"slot {r['slot']}, ant. {r['antenna']}", str(r["n"]), str(r["max_degree"])], [SURFACE] * 3, [INK] * 3
        for k, _, _, _, fmt, _, _, _ in RULES:
            s = r["status"][k]
            line.append(_fmt(fmt, r.get(k)))
            col.append(SURFACE if s is None or s == OK else _tint(STATUS_COLOR[s]))
            ik.append(INK if s != REVISE else "#a02020")
        line.append(STATUS_TEXT[r["verdict"]])
        col.append(STATUS_COLOR[r["verdict"]])
        ik.append(STATUS_INK[r["verdict"]])
        cells.append(line), colours.append(col), ink.append(ik)
    _table_figure(cells, colours, ink,
                  ["QUBO", "Atoms", "Max conflicts"] + [short for *_, short in RULES] + ["Verdict"],
                  [1.3, 0.5, 0.95] + [0.68] * len(RULES) + [0.8], "Embedding check per QUBO",
                  "Tinted cells: metric outside the rule of thumb (yellow: check, red: revise). Worst first.\n"
                  "Verdict: revise if the result changes (hard conflict broken, loss), check if only the structure "
                  "or the optimum before the prune is off.", path, row_h=0.26)


def plot_heatmaps(rows, path, max_rows=6):
    """
    c = 2Q_ij | s U_ij | difference for the worst QUBOs (revise, then check by error) and the
    largest OK one, capped at the largest weight (a coupling above it already forbids the pair).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    worst = sorted([r for r in rows if r["verdict"] != OK], key=lambda r: (-r["verdict"], -r["residual"]))
    good = [r for r in rows if r["verdict"] == OK]
    pick = worst[:max_rows - 1] + ([max(good, key=lambda r: (r["n"], r["edges"]))] if good else [])
    if not pick:
        return
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b42a2a"])
    fig, axes = plt.subplots(len(pick), 3, figsize=(10.5, 3.3 * len(pick)), facecolor=SURFACE, squeeze=False)
    for row_ax, r in zip(axes, pick):
        n, cap = r["n"], r["w_max"]
        iu = np.triu_indices(n, 1)
        mats = []
        for key in ("c", "su"):
            M = np.zeros((n, n))
            M[iu] = np.minimum(np.asarray(r[key]), cap)
            mats.append(M + M.T)
        D = mats[1] - mats[0]
        H = np.zeros((n, n), dtype=bool)
        H[iu] = r["hard"]
        H = H | H.T
        mask = np.eye(n, dtype=bool)
        for ax, M, cmap, norm, name in [
                (row_ax[0], mats[0], seq, None, "QUBO couplings  c = 2Q"),
                (row_ax[1], mats[1], seq, None, "Register  s·U"),
                (row_ax[2], D, div, TwoSlopeNorm(0, -cap, cap), "Difference  s·U − c")]:
            im = ax.imshow(np.ma.masked_where(mask, M), cmap=cmap, norm=norm,
                           vmin=None if norm else 0, vmax=None if norm else cap)
            ax.set_facecolor(SURFACE)
            if name != "Difference  s·U − c":
                for i, j in zip(*np.nonzero(np.triu(H))):
                    for a, b in ((i, j), (j, i)):
                        ax.text(b, a, "H", ha="center", va="center", fontsize=6 if n > 10 else 7,
                                color="#ffffff" if M[a, b] > 0.55 * cap else INK)
            ax.set_xticks(range(n), r["users"], fontsize=6 if n > 10 else 7, rotation=90, color=MUTED)
            ax.set_yticks(range(n), r["users"], fontsize=6 if n > 10 else 7, color=MUTED)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(name, fontsize=9, color=INK_2, loc="left")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
        row_ax[0].set_ylabel(f"slot {r['slot']}, antenna {r['antenna']}\n{n} atoms, error {r['residual']:.0%}\n"
                             f"{STATUS_TEXT[r['verdict']]}", fontsize=9, color=INK, labelpad=8)
    fig.suptitle("QUBO vs register, per pair of users (H: hard conflict; colours capped at the largest weight, "
                 "above which a coupling already forbids the pair)", fontsize=9, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.set_dpi(min(100, SCREEN_PX / fig.get_figheight()))   ## on screen: fit the height (the file keeps dpi 200)
    fig.savefig(path, dpi=200, facecolor=SURFACE)        ## left open: shown at the end if SHOW


if __name__ == "__main__":

    import logging
    logging.disable(logging.WARNING)

    QUBOS = os.path.join("results", "qubos_K120_C6_seed2.json")       ## from buildQUBO.py
    LIMIT = None                                                       ## None = all
    HUB_DEGREE, MAX_HUBS = 6, 2      ## hub split: users with more couplings are decided by branching
    REDUCED = True                   ## True: check the QUBO qubo-solver embeds (after variable fixing); False: the full QUBO
    HUB_POLICY = "defer"             ## None or "defer": as make_qubo_oracle(hub_policy=...) (needs REDUCED)
    SHOW = True                      ## True: also open the figures in windows (the script waits until they are closed)
    REUSE_FULL = False               ## True: take the whole-QUBO check from the last saved run (only the split is redone)
    OUT = QUBOS.replace("qubos_", "embedding_check_" + ("reduced_" if REDUCED else "")
                        + (f"{HUB_POLICY}_" if HUB_POLICY else ""))

    data = load_qubos(QUBOS)
    records = [r for r in data["problems"] if r["summary"]["variables"] >= 2]
    skipped = len(data["problems"]) - len(records)
    records = records[:LIMIT] if LIMIT else records
    print(f"{QUBOS}: {len(records)} QUBOs with >= 2 variables ({skipped} trivial skipped), "
          f"{'reduced by variable fixing (what qubo-solver embeds)' if REDUCED else 'full'}")

    rows, uncoupled, t0 = [], 0, time.perf_counter()
    if REUSE_FULL and os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        uncoupled = len(records) - len(rows)
        print(f"  whole-QUBO check reused from {OUT}")
    for i, rec in enumerate([] if rows else records):
        try:
            row = check_one(rec, REDUCED, HUB_POLICY)
            if row is None:
                uncoupled += 1
            else:
                rows.append(row)
        except Exception as e:                       ## keep going, report at the end
            rows.append({"slot": rec["slot"], "antenna": rec["antenna"], "n": rec["summary"]["variables"],
                         "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(records)} done ({time.perf_counter() - t0:.0f} s)", flush=True)

    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    R = lambda k: np.array([r[k] for r in ok], dtype=float)
    n, md = R("n"), R("max_degree")
    print(f"\n  No coupled pair (trivially embeddable, nothing to check): {uncoupled}")
    print(f"  Embedded: {len(ok)}/{len(rows)}   errors: {len(err)}   "
          f"atoms per register mean {R('atoms').mean():.1f}, max {R('atoms').max():.0f}"
          + (f" (first: {err[0]['error']})" if err else ""))
    print(f"  Residual ||sU - c|| / ||c||: median {np.median(R('residual')):.3f}, "
          f"90% {np.percentile(R('residual'), 90):.3f}, max {R('residual').max():.3f}   "
          f"(edges only: median {np.median(R('residual_edges')):.3f})")
    print(f"  Largest spurious coupling / median edge coefficient: median {np.median(R('spurious')):.3f}, "
          f"max {R('spurious').max():.3f}")
    rc = [r["rank_corr"] for r in ok if r["rank_corr"] is not None]
    to = [r["top20_overlap"] for r in ok if r["top20_overlap"] is not None]
    print(f"  Order of couplings (edges): rank correlation Q vs U median {np.median(rc):.2f}, "
          f"10% {np.percentile(rc, 10):.2f} ({len(rc)} QUBOs with >= 3 edges)   "
          f"strongest 20% kept strongest: mean {np.mean(to):.0%}")
    print(f"  Pairs without edge in Q but stronger in U than the weakest edge: "
          f"mean {R('non_edge_above_weakest').mean():.1%} of them, max {R('non_edge_above_weakest').max():.0%}")
    hk = [r["hard_ok"] for r in ok if r["hard_ok"] is not None]
    print(f"  Hard pairs still above max w after embedding: {sum(hk)}/{len(hk)} QUBOs with hard edges")
    print(f"  Ground state of the embedded QUBO: optimal for the original QUBO in "
          f"{sum(r['ground_optimal'] for r in ok)}/{len(ok)} (same bitstring: {sum(r['same_ground'] for r in ok)})   "
          f"feasible (no hard pair): {sum(r['feasible_emb'] for r in ok)}/{len(ok)}")
    gap = R("gap_emb")
    print(f"  After prune: embedded = optimum in {np.isclose(gap, 0, atol=1e-9).sum()}/{len(ok)}, "
          f"mean gap {gap.mean():.2%}, max {gap.max():.2%}   "
          f"(original QUBO + prune: {np.isclose(R('value_ref_pruned'), R('optimum')).sum()}/{len(ok)})")
    print(f"  Register: radial/min distance ratio max {R('dist_ratio').max():.2f} "
          f"(device allows {ok[0]['device_ratio']:.2f})   embed time mean {R('embed_s').mean():.1f} s")
    for lo, hi in [(0, 3), (3, 7), (7, 99)]:
        sel = (md >= lo) & (md < hi)
        if sel.any():
            g = np.array([r["same_ground"] for r in ok])[sel]
            print(f"    max degree {lo}-{hi - 1 if hi < 99 else '+'}: {sel.sum():3d} QUBOs, residual median "
                  f"{np.median(R('residual')[sel]):.3f}, same ground {g.mean():.0%}, "
                  f"mean gap after prune {gap[sel].mean():.2%}")

    ## Hub split: the QUBOs with a hub, each branch embedded on its own
    hub_rows = [] if REDUCED else [r for r in ok if r["max_degree"] > HUB_DEGREE]   ## split works on the full graph
    by_key = {(r["slot"], r["antenna"]): r for r in data["problems"]}
    split_rows, t1 = [], time.perf_counter()
    for i, r in enumerate(hub_rows):
        try:
            split_rows.append(check_split(by_key[(r["slot"], r["antenna"])], HUB_DEGREE, MAX_HUBS))
        except Exception as e:
            split_rows.append({"slot": r["slot"], "antenna": r["antenna"], "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 10 == 0:
            print(f"  hub split {i + 1}/{len(hub_rows)} done ({time.perf_counter() - t1:.0f} s)", flush=True)
    sp = [s for s in split_rows if "error" not in s]
    if sp:
        before = np.array([r["gap_emb"] for r in hub_rows if any((s["slot"], s["antenna"]) == (r["slot"], r["antenna"]) for s in sp)])
        after = np.array([s["gap"] for s in sp])
        print()
        print(f"  Hub split ({len(sp)} QUBOs with a user of more than {HUB_DEGREE} couplings, "
              f"at most {MAX_HUBS} hubs fixed):")
        print(f"    branches per QUBO mean {np.mean([s['branches'] for s in sp]):.1f}   atoms per branch max mean "
              f"{np.mean([s['atoms_max'] for s in sp]):.1f}   largest degree left mean "
              f"{np.mean([s['branch_max_degree'] for s in sp]):.1f}   errors {len(split_rows) - len(sp)}")
        print(f"    residual (branches, median) {np.median([s['residual_median'] for s in sp]):.3f}   "
              f"(whole QUBO: {np.median([r['residual'] for r in hub_rows]):.3f})")
        print(f"    optimum after prune: whole QUBO embedded {np.isclose(before, 0, atol=1e-9).mean():.0%}  ->  "
              f"split and embedded {np.isclose(after, 0, atol=1e-9).mean():.0%}   "
              f"mean gap {before.mean():.2%} -> {after.mean():.2%}, max {before.max():.1%} -> {after.max():.1%}")

        ## The three checks against the interaction matrix: whole hub QUBOs vs their branches
        br = [m for s in sp for m in s["branch_metrics"] if m.get("embedded")]
        def stats(items, key, how):
            v = np.array([x[key] for x in items if x.get(key) is not None], dtype=float)
            return how(v) if v.size else float("nan")
        lines = [
            ("residual ||sU - c||/||c||, median", "residual", np.median, "{:.3f}"),
            ("residual, 90%", "residual", lambda v: np.percentile(v, 90), "{:.3f}"),
            ("largest spurious / median edge, median", "spurious", np.median, "{:.3f}"),
            ("largest spurious / median edge, max", "spurious", np.max, "{:.2f}"),
            ("rank correlation Q vs U, median", "rank_corr", np.median, "{:.2f}"),
            ("rank correlation, 10%", "rank_corr", lambda v: np.percentile(v, 10), "{:.2f}"),
            ("strongest 20% kept strongest", "top20_overlap", np.mean, "{:.0%}"),
            ("no-edge pairs above weakest edge", "non_edge_above_weakest", np.mean, "{:.1%}"),
            ("hard pairs still above max w", "hard_ok", np.mean, "{:.0%}"),
            ("embedded ground state optimal", "ground_optimal", np.mean, "{:.0%}"),
        ]
        print(f"    Against the interaction matrix: {len(hub_rows)} whole hub QUBOs vs {len(br)} embedded branches")
        print(f"      {'':40s} {'whole QUBO':>11s} {'branches':>9s}")
        for label, key, how, fmt in lines:
            print(f"      {label:40s} {fmt.format(stats(hub_rows, key, how)):>11s} {fmt.format(stats(br, key, how)):>9s}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"qubos": QUBOS, "rows": rows, "hub_split": split_rows}, f)
    print(f"  Saved {OUT}")

    try:
        import matplotlib
        if not SHOW:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        same = np.array([r["same_ground"] for r in ok])
        for flag, col, lab in [(True, "#2a7ab9", "same ground state"), (False, "#d1495b", "ground state changed")]:
            ax[0].scatter(md[same == flag], R("residual")[same == flag], s=14, c=col, label=lab, alpha=0.8)
            ax[1].scatter(n[same == flag], R("residual")[same == flag], s=14, c=col, alpha=0.8)
        ax[0].set_xlabel("largest node degree"); ax[0].set_ylabel(r"relative residual $\|sU-c\|/\|c\|$")
        ax[1].set_xlabel("variables (users)")
        ax[0].legend(frameon=False, fontsize=8)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT.replace(".json", ".png"), dpi=200)
        print(f"  Saved {OUT.replace('.json', '.png')}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    ## Cost of the defer itself (no embedding): exact solution without the hubs, hubs added back, prune
    if HUB_POLICY == "defer":
        cost = []
        for rec in records:
            G, w, cap = problem_graph(rec)
            G_q, deferred = defer_hubs(G, w)
            if not deferred:
                continue
            q = build_antenna_qubo(G_q, w, None)
            fixed, _, shift, live = hansen_recursive(q["Q"])
            Qr = q["Q"][np.ix_(live, live)] + np.diag(shift[live])
            S = drop_hard(G, w, decode(q, lift(brute_ground(Qr), live, fixed, q["Q"].shape[0])))
            S_final = prune(G, w, add_back(G, w, S, deferred), cap)
            opt = rec["reference"]["optimum_value"]
            cost.append({"slot": rec["slot"], "antenna": rec["antenna"], "deferred": deferred,
                         "added_back": [h for h in deferred if h in S_final], "atoms": len(live),
                         "loss": (opt - antenna_objective(G, w, S_final)) / opt if opt > 0 else 0.0})
        print(f"\n  Defer (exact, no embedding): {len(cost)} QUBOs with a hub, "
              f"{sum(len(c['deferred']) for c in cost)} hubs left out, "
              f"{sum(len(c['added_back']) for c in cost)} added back to the same antenna")
        for c in cost:
            print(f"    slot {c['slot']}, antenna {c['antenna']}: hub {c['deferred']} "
                  f"{'added back' if c['added_back'] == c['deferred'] else 'NOT added back'}, "
                  f"{c['atoms']} atoms left, loss {c['loss']:.1%}")

    ## Report: status per metric and per QUBO, tables and heatmaps
    judged = assess([r for r in ok if r["edges"] >= 2 and "c" in r])   ## one coupling: fitted exactly
    lines = summary_table(judged)
    print(f"\n  Report ({len(judged)} QUBOs with >= 2 couplings, {len(ok) - len(judged)} with one coupling left out):")
    print(f"    {'Metric':52s} {'Ideal':6s} {'Good enough':22s} {'Yours':32s} {'OK':7s} Status")
    for ln in lines:
        print(f"    {ln['metric']:52s} {_plain(ln['ideal']):6s} {_plain(ln['good']):22s} {ln['yours']:32s} "
              f"{ln['ok']:7s} {STATUS_PLAIN[ln['status']]}")
    for r in sorted(judged, key=lambda r: -r["verdict"]):
        if r["verdict"] != OK:
            bad = [short for k, *_, short in RULES if r["status"][k] not in (None, OK)]
            print(f"    {STATUS_PLAIN[r['verdict']]:7s} slot {r['slot']}, antenna {r['antenna']} "
                  f"({r['n']} atoms, max conflicts {r['max_degree']}): {', '.join(bad)}")
    try:
        import matplotlib
        if not SHOW:
            matplotlib.use("Agg")
        base = OUT.replace(".json", "")
        plot_summary(lines, len(judged), len(ok) - len(judged), [r["verdict"] for r in judged], base + "_summary.png")
        plot_per_qubo(judged, base + "_per_qubo.png")
        plot_heatmaps(judged, base + "_heatmaps.png")
        print(f"  Saved {base}_summary.png, _per_qubo.png, _heatmaps.png")
    except Exception as e:
        print(f"  (report plots skipped: {type(e).__name__}: {e})")

    import matplotlib.pyplot as plt
    if SHOW:
        plt.show()
    plt.close("all")
