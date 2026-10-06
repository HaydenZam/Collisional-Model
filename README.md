# Collision-Model

Collision model of an $N$-site spin-chain channel driven by a sender ancilla $S$ on
site 1 and a receiver ancilla $R$ on site $N$, whose joint state $\rho_{SR}$ can be
correlated (classically, discordant or entangled). The code compares the exact repeated
collisions with their master-equation limit, and asks when the resulting dissipation can
be unravelled bath by bath.

The package `colmodel/` holds every piece of physics once; `studies/` holds short,
seeded scripts that set parameters and plot; `tests/validate.py` checks the package
against independent calculations. The code before this reorganisation is the git tag
`pre-cleanup`.

## The model

Chain: $H_C=\sum_j \omega_j n_j+\sum_j J_j(\sigma^+_j\sigma^-_{j+1}+\mathrm{h.c.})$.
Each collision lasts $\tau$ with

$$H=H_C+\omega_S n_S+\omega_R n_R+\frac{g_L}{\sqrt\tau}(\sigma^-_S\sigma^+_1+\sigma^+_S\sigma^-_1)+\frac{g_R}{\sqrt\tau}(\sigma^-_N\sigma^+_R+\sigma^+_N\sigma^-_R),$$

and a fresh copy of $\rho_{SR}$ is used every collision:
$\rho_C\to\mathrm{tr}_{SR}[U(\rho_{SR}\otimes\rho_C)U^\dagger]$, $U=e^{-iH\tau}$ (`colmodel.collision`).

As $\tau\to0$ this becomes (`colmodel.master_equation`)

$$\dot\rho=-i[H_C+H_1,\rho]+\sum_{ab}D_{ab}\Big(A_a\rho A_b^\dagger-\tfrac12\{A_b^\dagger A_a,\rho\}\Big),\qquad A=(\sigma^+_1,\sigma^-_1,\sigma^+_N,\sigma^-_N),$$

* drive: $H_1=\frac{g_L}{\sqrt\tau}(\langle\sigma^-_S\rangle\sigma^+_1+\langle\sigma^+_S\rangle\sigma^-_1)+\frac{g_R}{\sqrt\tau}(\langle\sigma^-_R\rangle\sigma^+_N+\langle\sigma^+_R\rangle\sigma^-_N)$, finite only if the ancilla coherences scale as $\sqrt\tau$;
* Kossakowski matrix: $D=\begin{pmatrix}D_{LL}&D_{LR}\\D_{LR}^\dagger&D_{RR}\end{pmatrix}$, $D_{LL}=g_L^2\,\mathrm{diag}(p_s,1-p_s)$, $D_{RR}=g_R^2\,\mathrm{diag}(p_r,1-p_r)$, $D_{LR}=g_Lg_R\begin{pmatrix}J_{-+}&J_{--}\\J_{++}&J_{+-}\end{pmatrix}$, with $J_{xy}=\langle\sigma^x_S\sigma^y_R\rangle$.

$D$ is built from the raw second moments $J=\langle ab\rangle$. `kossakowski(..., connected=True)`
uses $\langle ab\rangle-\langle a\rangle\langle b\rangle$ instead; the difference is exactly
$-\tfrac12[H_1',[H_1',\rho]]$ with $H_1'$ the unscaled drive (checked in the tests). It only
matters for ancillas with local coherence: with $J$, an uncorrelated product ancilla
$|+\rangle|+\rangle$ gets $D_{LR}\neq0$ and a monitorable fraction of 0.

## Conventions

$|0\rangle$ = ground, $|1\rangle$ = excited; $\sigma^+=|1\rangle\langle0|$, $\sigma^-=|0\rangle\langle1|$,
$n=|1\rangle\langle1|$, $\sigma^z=|1\rangle\langle1|-|0\rangle\langle0|$ (excited $=+1$),
$\sigma^\pm=(\sigma^x\pm i\sigma^y)/2$. Registers are ordered left to right with `np.kron`;
collision space is $S\otimes C_1\cdots C_N\otimes R$. Coherence parameters are always
$c=\langle\sigma^-\rangle=\rho_{10}$. Arrays are numpy throughout; qutip is used only
inside `collision.py` and `master_equation.py`. Superoperators act on column-stacked
vectors (`master_equation.vec` / `unvec`).

## Layout

| Path | Contents |
|---|---|
| `colmodel/operators.py` | conventions, single-qubit operators, `kron`, `embed`, `ptrace`, state checks |
| `colmodel/ancilla.py` | S-R states: representatives (product, classical, discord, entangled), Bell-diagonal, thermal-correlated, coherent product, random (seeded), $\sqrt\tau$ scaling |
| `colmodel/correlations.py` | negativity, concurrence, discord, mutual information, local coherence, `classify` |
| `colmodel/model.py` | `Params` (shared by both dynamics), chain and collision Hamiltonians |
| `colmodel/collision.py` | exact repeated collisions |
| `colmodel/master_equation.py` | ancilla moments, Kossakowski $D$, drive, Liouvillian, `solve` (mesolve or exact expm); generic GKSL `superoperator`, `evolve` |
| `colmodel/unravelling.py` | GKSL test, strict test, `monitorable_fraction` (bisection or generalised eigenvalue), Schur and closed-form cross-checks, canonical channels and jump operators |
| `colmodel/trajectories.py` | quantum-jump Monte Carlo: readable single run, vectorised ensemble, Kraus instrument check |
| `colmodel/toy.py` | two-qubit model $c=\gamma\begin{pmatrix}1&m\\m&1\end{pmatrix}$ and its vacuum collision-model derivation |
| `colmodel/metrics.py` | trace distance, site populations, error report |
| `studies/` | one script per question, output to `runs/<study>/`; index in `studies/README.md` |
| `tests/validate.py` | 25 checks against independent calculations (about 10 s) |
| `runs/` | output, git-ignored |

## Running

```bash
conda activate quantum-ml
pip install -e .                      # once
python tests/validate.py              # all checks
python studies/collision_vs_me.py --ancilla entangled
```

Every study takes `--help`. `environment.yml` pins the versions in `quantum-ml`.

## Notes

* **Convergence of the collision model to the master equation.** For Bell-diagonal,
  thermal-correlated and $\sqrt\tau$-scaled product ancillas the trace distance at fixed time is
  $O(\tau)$. For ancillas with mixed correlations such as $\langle\sigma^z_S\sigma^x_R\rangle\neq0$
  (e.g. `ancilla.random_state`, and therefore the `all` mixture) it is only $O(\sqrt\tau)$:
  the third-order term of each collision is $O(\tau^{3/2})$ and involves three-operator
  moments such as $\langle n_S\,\sigma^x_R\rangle=\tfrac12\langle\sigma^z_S\sigma^x_R\rangle$,
  which vanish for Bell-diagonal states but not here. The master equation is still the limit,
  but the error is larger: at $\tau=10^{-3}$, $T=2$, $N=2$, the trace distance is
  $2.7\times10^{-5}$ for `random_state(seed=5)` against $1.7\times10^{-6}$ for the Bell state.
  Both rates are checked in the tests.
* **"no D_LR".** `collision_vs_me.py --no-nonlocal` drops $D_{LR}$ from the master equation;
  for the Bell-state ancilla the trace distance at $T=10$ goes from $6\times10^{-6}$ to 0.14.
* **Random states and the sign convention.** `ancilla.random_state(seed)` gives the state the
  old `random_two_qubit_state(seed=...)` gave, conjugated by $\sigma^x\otimes\sigma^x$ (both
  qubits flipped), because $\sigma^y,\sigma^z$ changed sign. Correlation measures and
  monitorable fractions are unchanged; populations become $1-p$.
