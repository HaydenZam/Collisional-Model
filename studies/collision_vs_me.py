"""Exact collision model vs its master equation, for one sender-receiver ancilla state.

Runs the repeated collisions and the master equation from the all-ground chain and
compares them: trace distance between the two chain states, and the per-site
populations <n_j> (sigma^z_j = 2 n_j - 1).

Ancillas (as in the original Collision_Model.py):
  product     |+>|+> with its coherences scaled by sqrt(tau)
  classical, discord, entangled   the zero-coherence representatives
  random      random zero-coherence state (--seed)
  sum         (product + entangled) / 2
  all         mean of product, classical, discord, entangled and random
  coherent    product of coherent qubits: excited populations --ps, --pr and drive
              amplitudes --cs, --cr = <sigma^->/sqrt(tau). The defaults are the
              archived 2_finite_temp.py run; "--ps 0.5 --cs 0.5 --pr 0 --cr 0" is
              the archived Carola.py setup (coherent sender, ground-state receiver).

--no-nonlocal drops the non-local block D_LR from the master equation (the
"no D_LR" plots of the pre-cleanup snapshot).

Examples
  python studies/collision_vs_me.py --ancilla entangled
  python studies/collision_vs_me.py --ancilla entangled --no-nonlocal
  python studies/collision_vs_me.py --ancilla coherent --N 3 --tau 5e-4

Output: runs/collision_vs_me/<label>.png and .npz (times, trace distance, populations).
"""

import argparse

import _common  # noqa: F401  (puts the repo on sys.path)
import numpy as np

from colmodel import Params, ancilla, collision, correlations, master_equation as me, metrics

CHOICES = ("product", "classical", "discord", "entangled", "random", "sum", "all", "coherent")


def make_ancilla(args):
    tau = args.tau
    product = ancilla.collision_scaled_product(ancilla.representative("product"), tau)
    if args.ancilla == "product":
        return product
    if args.ancilla in ("classical", "discord", "entangled"):
        return ancilla.representative(args.ancilla)
    if args.ancilla == "random":
        return ancilla.random_state(seed=args.seed)
    if args.ancilla == "sum":
        return 0.5 * (product + ancilla.representative("entangled"))
    if args.ancilla == "all":
        parts = [product] + [ancilla.representative(n) for n in ("classical", "discord", "entangled")]
        parts.append(ancilla.random_state(seed=args.seed))
        return sum(parts) / 5
    if args.ancilla == "coherent":
        s = np.sqrt(tau)
        return ancilla.coherent_product(args.ps, s * args.cs, args.pr, s * args.cr)
    raise ValueError(args.ancilla)


def label(args):
    name = args.ancilla
    if args.ancilla in ("random", "all"):
        name += f"_seed{args.seed}"
    if args.ancilla == "coherent":
        name += f"_ps{args.ps:g}_cs{args.cs:g}_pr{args.pr:g}_cr{args.cr:g}".replace("(", "").replace(")", "")
    if args.no_nonlocal:
        name += "_noDLR"
    return f"{name}_N{args.N}_tau{args.tau:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ancilla", choices=CHOICES, default="all")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--J", type=float, default=0.3)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--gL", type=float, default=0.2)
    ap.add_argument("--gR", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--n-steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="for the random state (ancilla random/all)")
    ap.add_argument("--ps", type=float, default=0.7)
    ap.add_argument("--cs", type=complex, default=complex(0.4, -0.3))
    ap.add_argument("--pr", type=float, default=0.4)
    ap.add_argument("--cr", type=complex, default=complex(0.3, -0.5))
    ap.add_argument("--no-nonlocal", action="store_true")
    ap.add_argument("--method", choices=("mesolve", "expm"), default="mesolve")
    ap.add_argument("--show", action="store_true", help="also open the figure window")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Params(N=args.N, J=args.J, omega=args.omega, gL=args.gL, gR=args.gR, tau=args.tau)
    rho_SR = make_ancilla(args)
    out = _common.out_dir("collision_vs_me")
    name = label(args)

    # ---- ancilla summary -------------------------------------------------
    m = me.ancilla_moments(rho_SR)
    np.set_printoptions(precision=4, suppress=True)
    print(f"ancilla: {args.ancilla}   (N = {p.N}, tau = {p.tau:g}, {args.n_steps} collisions, T = {p.tau*args.n_steps:g})")
    print(rho_SR)
    print(f"concurrence = {correlations.concurrence(rho_SR):.4f}   discord = {correlations.discord(rho_SR):.4f}"
          f"   p_s = {m['p_s']:.4f}   p_r = {m['p_r']:.4f}")
    print("means <s-_S>, <s+_S>, <s-_R>, <s+_R> =", m["means"])
    print("J (mm, mp, pm, pp) =", np.array([m["J"][k] for k in ("mm", "mp", "pm", "pp")]))
    print("C (mm, mp, pm, pp) =", np.array([m["C"][k] for k in ("mm", "mp", "pm", "pp")]))

    # ---- dynamics ----------------------------------------------------------
    coll = collision.run_collisions(p, rho_SR, args.n_steps)
    mes = me.solve(p, rho_SR, coll.times, nonlocal_terms=not args.no_nonlocal, method=args.method)
    td = metrics.trace_distance(mes, coll.states)
    pop_c, pop_m = metrics.populations(coll.states, p.N), metrics.populations(mes, p.N)

    print(f"\ntrace distance: at T = {td[-1]:.4e}, max = {td.max():.4e}")
    for j in range(p.N):
        metrics.error_report(pop_c[j], pop_m[j], name=f"<n_{j+1}> collision vs ME")

    np.savez(out / f"{name}.npz", times=coll.times, trace_distance=td,
             populations_collision=pop_c, populations_me=pop_m, rho_SR=rho_SR)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(coll.times, td, lw=2)
    ax[0].set_ylabel("trace distance")
    title = f"{p.N}-site channel: {args.ancilla}" + (" (no D_LR)" if args.no_nonlocal else "")
    ax[0].set_title(title)
    for j in range(p.N):
        line, = ax[1].plot(coll.times, pop_c[j], lw=2, label=rf"collision $\langle n_{j+1}\rangle$")
        ax[1].plot(coll.times, pop_m[j], "--", lw=2, color=line.get_color(), label=rf"ME $\langle n_{j+1}\rangle$")
        ax[2].plot(coll.times, pop_c[j] - pop_m[j], lw=2, label=rf"$\Delta\langle n_{j+1}\rangle$")
    ax[1].set_ylabel("population")
    ax[1].legend(fontsize=8)
    ax[2].axhline(0.0, ls="--", lw=1, color="k")
    ax[2].set_ylabel("collision - ME")
    ax[2].set_xlabel("time")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    print(f"\nwrote {out / name}.png and .npz")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
