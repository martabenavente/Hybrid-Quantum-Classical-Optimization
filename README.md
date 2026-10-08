# Hybrid quantum-classical optimisation for Cell-Free Massive MIMO

MSc in Quantum Computing project (VPIMA801), in collaboration with
[Pasqal](https://www.pasqal.com/).

The project applies hybrid quantum-classical optimisation to resource allocation in Cell-Free communication networks. A classical pipeline generates realistic scenarios, builds graph representations of the allocation problem, and checks the resulting schedules. The hard graph subproblems are written as QUBOs to be solved on Pasqal's emulator and, when available, on neutral atom hardware. The full project description is in [`SDU docs/VPIMA801_Project_description.pdf`](SDU%20docs/VPIMA801_Project_description.pdf)..

## Current formulation: users to antennas, one antenna at a time

Every timeslot, each user is assigned to one antenna.

- **Antennas.** Each antenna has its own 20 MHz band, split into `C`
  channels, so it serves at most `C` users per slot. A user may only use
  antennas within range (300 m) that are among its 3 strongest.
- **One small graph per antenna and timeslot.** The nodes are the users still
  unassigned that the antenna can serve. Each antenna graph is a small
  weighted problem, sized to fit the emulator (about 20–25 nodes).
- **Node weight.** Link quality plus a continuity bonus:
  `w_u = SE(u, a) / 5.55 + 0.5 · [a served u in the previous slot]`.
  Spectral efficiency (SE) is capped at 5.55 bits/s/Hz (uplink 64QAM).
- **Edges are constraints.**
  - *Hard*: users closer than 25 m (blockage) can never share an antenna.
  - *Soft*: adjacent-channel leakage. The edge cost `J_uv` is the SE the pair is expected to lose by sharing the antenna. Two
    users are on adjacent channels with probability `2 / C`.
- **Per-antenna objective.** `max Σ w_u x_u − Σ J_uv x_u x_v`, with no hard pair and at most `C` users.
  
### One timeslot

1. **Order the antennas.** Antennas that served the most users in the
   previous slot go first, then those with the most users in range.
2. **Build the antenna graph** from the users still unassigned.
3. **Oracle.** Solve the per-antenna problem. The oracle is pluggable: a greedy
   marginal gain oracle is the classical baseline, and tabu search, simulated
   annealing or a quantum sampler can replace it.
4. **Prune to `C`.** If the oracle returns too many users, drop the one whose
   removal hurts the objective least.
5. **Repair coverage.** Users left out are served by a min-cost max-flow that
   moves as few already assigned users as possible.
   
### QUBO - ON THE WAY

### Scenario

- 1 km² area with 30 antennas and uplink at 100 mW.
- Path loss and correlated shadowing come from the Cell-Free simulator in `functionsSetup.py`.
- The scenario is dynamic: between consecutive slots, 20–50% of the users
  change. Half are replaced by new users and half move up to 90 m.
  
## Repository layout

| Path | Contents |
|---|---|
| `Code/functionsSetup.py` | Network setup: antenna and user positions, path loss, shadowing, pilots |
| `Code/generate_scenario.py` | Static scenario data and interference matrix |
| `Code/dynamic_scenario.py` | `DynamicScenario`: users arrive, leave and move between timeslots |
| `Code/buildConflictGraph.py` | Per-antenna graphs: candidates, compatibility, hard and soft constraints |
| `Code/greedySolver.py` | Greedy oracle, prune, coverage repair and the multi-slot scheduler |
| `Code/buildQUBO.py` | QUBO formulation of the per-antenna problem |
| `Code/evaluate.py` | Validity checks and metrics: SE, coverage, rate distribution, Jain index, handovers |
| `QUBO_try.py` | Minimal test of Pasqal's `qubo-solver` |
| `Meeting follow ups/` | Weekly items reviews |
| `Bibliography/` | Reference papers, including arXiv:2301.02637 |
| `SDU docs/` | Project description |

## Getting started

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
cd Code
python greedySolver.py   # classical pipeline over T = 10 slots, with the evaluation report
python buildQUBO.py      # per-antenna QUBOs, checked by enumeration and saved to results/
```

Scenario, antenna, weight and constraint parameters are set at the bottom of
each script (`K`, `L`, `T`, `CAPACITY`, `ALPHA_*`, `MIN_SEP_M`, `LEAKAGE_DB`,
and so on). The QUBO oracles also need `torch` and Pasqal's `qubo-solver`.

### Benchmarking plan

Every oracle solves the same scenario and seed, and every schedule is checked with the same validity rules:
- the greedy oracle (baseline);
- the QUBO solved classically (tabu search, simulated annealing);
- Pulser-based quantum sampling on the emulator or QPU.

Metrics are the gap to the exact optimum per QUBO, total spectral efficiency, coverage (must be 100%), worst, 10th-percentile and median user rate, Jain index, handovers, and runtime against QUBO size.

## References

- W. da Silva Coelho, L. Henriet, L.-P. Henry. *A quantum pricing-based column
  generation framework for hard combinatorial problems*. arXiv:2301.02637,
  2023.
- A. Lucas. *Ising formulations of many NP problems*. Frontiers in Physics,
  2014.
- Ö. T. Demir, E. Björnson, L. Sanguinetti. *Foundations of User-Centric
  Cell-Free Massive MIMO*. 2023.
