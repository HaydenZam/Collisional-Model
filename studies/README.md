# Studies

Each script answers one question, takes its parameters on the command line (`--help`),
is seeded, and writes to `runs/<script name>/` (git-ignored). Run from the repository root,
e.g. `python studies/collision_vs_me.py --ancilla discord`. Times are on the Ubuntu machine.

| Script | Question | Output | Time | Came from |
|---|---|---|---|---|
| `collision_vs_me.py` | How close is the master equation to the exact collision model, for a given S-R ancilla? Options: `--ancilla`, `--N`, `--tau`, `--n-steps`, `--no-nonlocal`, coherent-ancilla parameters | `<label>.png` (trace distance, populations, differences), `<label>.npz` | ~10 s | `Collision_Model.py`, `Archive/2_finite_temp.py`, `Archive/Carola.py` |
| `ancilla_table.py` | Coherence, entanglement, discord, class and $\|D_{LR}\|$ of the ancilla states | `table.csv` | ~10 s | `random_two_qubit_state.py` main, `Unravelling_Test.py` |
| `unravelling_landscape.py` | How much of one bath can be monitored, and what sets it? | `named_reps.csv`, `fraction_landscape.png`, `c3_invariance.png`, `frac_vs_correlation.png`, `thermal_marginals.png`, `log.txt` | ~30 s | `Unravel.py` main |
| `two_qubit_unravelling.py` | When does a per-bath unravelling exist? (toy model, trajectories, Kraus check, energy ledger, vacuum derivation, partial efficiency) | `report.txt` | ~30 s | `unraveling_example.py` main |
| `trajectory_figures.py` | What the trajectories look like, for valid and invalid unravellings | `trajectories_anatomy.png`, `valid_vs_invalid.png` | ~35 s | `plot_trajectories.py` |

## Reproducing the pre-cleanup outputs

* `Plots/Non-Separable/M_vs_E_<x>.jpeg`: `collision_vs_me.py --ancilla <x>` (x = entangled, sum, all).
  `all` used an unseeded random state, so only its seeded version can be regenerated.
* `... no D_LR.jpeg`: add `--no-nonlocal`.
* `Plots/Separable/M_vs_E_{product,classical,discord}.jpeg`: `--ancilla product|classical|discord`.
* `Plots/Separable/M_vs_E_ps-0.7_cs-(0.4+0.3j)_pr-0.4_cr-(0.3+0.5j).jpeg`: `--ancilla coherent`
  (defaults; the old file named $\rho_{01}=\langle\sigma^+\rangle$, the new parameters are
  $\langle\sigma^-\rangle/\sqrt\tau$, hence the conjugated values 0.4-0.3j, 0.3-0.5j).
* `unravelling_out/*`: `unravelling_landscape.py`. Only change: `thermal_marginals.png` has a
  corrected right-hand title (the fraction depends on the distance of $p_s$ from 1/2, and is
  symmetric under $p_s\to1-p_s$).
* Trajectory figures and the toy-model report are reproduced exactly (same seeds, same random
  stream).
