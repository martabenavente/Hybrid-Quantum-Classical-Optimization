# Summary: preprocessing, embedding check and hub defer (2026-10-29)

Context for continuing this work. Project VPIMA801: assigning users to antennas
(one antenna per user), solved antenna by antenna. Each antenna gives one graph,
hence one QUBO, solved with Pasqal's **qubo-solver**. The assignment loop is
`assign_users_to_antennas` (greedySolver.py); the QUBO oracle is
`make_qubo_oracle` (solveQUBO.py).

## Setup

- **QUBO per antenna** (buildQUBO.py):
  E(x) = - Σ w_u x_u + Σ_soft J_uv x_u x_v + P Σ_hard x_u x_v
  - w_u: link quality + continuity (whether the antenna already served u in the previous slot)
  - J_uv: expected leakage between users (soft conflict)
  - Hard conflict: users < 25 m apart, penalty P = 1.5·w_max
  - Capacity is not in the QUBO: the prune step of the assignment enforces it.
- **Test data:** results/qubos_K120_C6_seed2.json. K=120 users, L=30 antennas,
  T=10 slots, seed 2, capacity 6. That gives 262 antenna QUBOs with 1884 variables.
- **qubo-solver "quantum" method:**
  1. Preprocessing: recursive Hansen variable fixing.
  2. Neutral-atom emulator. Device AnalogDeviceWithDMM, BLaDE embedder,
     local_energy_scale drive shaper.
  3. Postprocessing: one sweep of single-bit-flip local search.

## 1. Preprocessing: variables fixed to 1, and how their neighbours are affected

When preprocessing fixes a variable to 1, its atom is removed. Each neighbour
should then receive the removed coupling (2Q_uv = J_uv) as a shift on its
diagonal. Two options were considered for getting that shift onto the hardware:
(a) keep the fixed atom in the register; (b) pass the shift on through local
detuning (DMM).

Findings (checkFixing.py, verified against qubo-solver's own fixing on 262/262 QUBOs):

| | |
|---|---|
| Fixed to 1 | 1547 of 1884 variables (82%) |
| … with neighbours that stay as atoms (the case that matters) | 224, in 57 QUBOs |
| Fixed to 0 | 65, all because a fixed-to-1 neighbour's shift pushed them there |
| QUBOs solved entirely by preprocessing | 184 of 262; 78 go to the hardware |
| Atoms that receive a shift | 161 of 272 (59%) |
| Size of the shift relative to the atom's own weight | median 19%, up to 99% |
| Shift lost (neither a nor b) | wrong ground state in 8 of the 57 QUBOs |
| Option (a) | +224 atoms (affected registers about 2× larger); the kept atom can still end in 0 |

**Decision: option (b).** qubo-solver already implements it. The embedder and
the drive shaper receive the *reduced* matrix, whose diagonal includes the
shift, and the drive shaper turns that diagonal into per-atom DMM detuning.
This needs a device with a DMM, so solveQUBO.py now warns if the device has none.

Emulator test (pulse only, no postprocessing), share of shots on the optimum:
with the shift 72%, without 62%. In the 8 QUBOs where the shift changes the
optimum: 64% vs 12%.

(Small issue in the pasted note: both options are labelled "(a)"; the second should be "(b)".)

## 2. Embedding check: QUBO vs the register's interaction matrix U

The three checks requested (checkEmbedding.py):

1. Global comparison of Q and U: relative error ||sU − c|| / ||c||, where
   c = 2Q_ij and s is one fitted scale; plus fake couplings (U on pairs with no conflict).
2. Order of the coefficients: rank correlation; whether the strongest 20% of
   couplings stay the strongest; no-conflict pairs that come out stronger than
   the weakest real conflict.
3. Solve the QUBO built from U and apply its solution to the original QUBO:
   is it optimal, are hard conflicts kept, and what is the loss after the capacity prune.

**Key point: check the reduced QUBO** (after fixing, shift included), because
that is what qubo-solver actually embeds. The earlier check used the full QUBO
and looked much worse. On the same 78 QUBOs:

| | Full QUBO | Reduced QUBO |
|---|---|---|
| Error Q vs U, median | 14% | 0% |
| Rank correlation, median | 0.81 | 0.96 |
| Hard conflicts still forbid the pair | 81% | 97% |
| U-QUBO solution optimal for the original | 44/78 | 73/78 |
| Optimal after prune | 59/78 | 70/78 (the exact QUBO gets 72) |

43–45 of the reduced QUBOs have a single coupling, which one fitted scale
reproduces exactly. They are excluded from the report so the medians are not
inflated.

## 3. Hubs and the "defer" policy

- **Hub:** a user who, after fixing, still conflicts with more than 6 others
  (e.g. slot 2 antenna 16: user 56 is linked to 12 of 15 users). A 2D register
  cannot reproduce such a star: some of its couplings come out too weak and
  fake couplings appear. All the embedding failures were hub QUBOs.
- **Defer** (`hub_policy="defer"` in make_qubo_oracle). Still one QUBO per antenna:
  1. Leave the hub out of the antenna's QUBO, then solve.
  2. Add the hub back if it has no hard conflict with the chosen users and a positive gain.
  3. Otherwise it stays unassigned: later antennas or the coverage repair handle it.
- **Truncate** (keep only a hub's 6 strongest couplings) was rejected. It fails
  on slot 6 antenna 3, where the hub's hard conflicts cannot be dropped.
- **Hub split** into several QUBOs per antenna was rejected: the design is one QUBO per antenna.

Results on the 4 hub QUBOs (2,16), (6,3), (5,17), (8,17):

| | Without defer | With defer |
|---|---|---|
| Full quantum method, final loss | 0% (the postprocessing rescues it) | 0% |
| Shots on the optimum, slot 2 ant. 16 | 33–56% (15 atoms) | 89% (2 atoms) |
| Shots on the optimum, slot 6 ant. 3 | 2–4% (10 atoms) | 16–18% (5 atoms) |
| Embedding: QUBOs to revise | 2 | 0 |
| Embedding loss, worst | 7.8% | 0.0% |

None of the 4 hubs belongs in its antenna's optimal solution anyway: their total
leakage (4.2–8.2) exceeds their own weight (1.0–1.5). So defer is exact in these
cases, and the hubs move on to other antennas, as they would without defer.
Only 4 cases were tested.

## 4. Current state of the embedding report (with defer, 32 non-trivial QUBOs)

- **Verdict: 20 OK, 12 check, 0 revise.**
  - revise = the result changes (hard conflict broken, loss ≥ 1%, or the register doesn't fit the device)
  - check = only the structure metrics are off
- Result metrics are all fine: hard conflicts 27/27, loss 0% in 32/32, optimal 30–31/32.
- Several "check"s are **false alarms**: tied couplings (several at 2.25) make
  "top 20%" and the rank correlation unstable, and "no-conflict" compares
  against tiny real couplings. Affected: slot 4 ant. 6, 9/18, 6/18, 5/5, 7/18.
- Real but harmless structural deviations: slot 6 ant. 10, 7/17 and 6/17 (users
  with 4–6 conflicts, some couplings close to 0 in the register); smaller ones
  in 3/17, 5/3 and 2/7.
- The embedder is random: numbers vary slightly between runs, verdicts so far have not.
- "Good enough" thresholds are rules of thumb (`RULES` in checkEmbedding.py), not standards.

## Code changes (uncommitted)

- **checkFixing.py (new):** replicates qubo-solver's fixing and reports how
  often variables are fixed and which neighbours are affected; optional
  emulator test with and without the shift.
- **checkEmbedding.py:**
  - `REDUCED` (check the reduced QUBO, default True), `HUB_POLICY` (default
    "defer"), `SHOW` (open the figures in windows).
  - Writes results/embedding_check_reduced_defer_K120_C6_seed2_{summary,per_qubo,heatmaps}.png
    and a .json. Runtime 5–10 min.
- **solveQUBO.py:** DMM warning; `make_qubo_oracle(hub_policy=None|"defer"|"truncate")`
  with helpers `reduced_hubs`, `defer_hubs`, `add_back`, `truncate_hubs`.
  The default is still None: the pipeline does not use defer yet.

## Open items

1. **Report fixes, pending approval:**
   - structure metrics at most yellow (red only for result problems)
   - treat couplings within about 5% as tied in "top 20%" and the rank correlation
   - "no-conflict": count only fake couplings above about 10% of a typical real coupling
   - show the result (loss, optimal) in each heatmap row's label
2. Turn on defer in the pipeline (`make_qubo_oracle(..., hub_policy="defer")`)
   and run the full assignment over all slots; check where the deferred users end up.
3. Optionally run the embedder several times per QUBO for stable numbers.
4. Optionally lower the hub threshold from 6 to 5 conflicts (would cover slot 6
   antenna 10); only if it causes a loss in a real run.
5. In the report: mention the 2D-register limitation for users with many conflicts.
