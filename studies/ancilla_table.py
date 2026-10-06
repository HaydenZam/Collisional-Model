"""Properties of the ancilla states: local coherence, entanglement, discord,
correlation class, and the size of the non-local Kossakowski block D_LR.

Also prints the Kossakowski matrix of the collision-scaled product state with
g = 1 (what the original Unravelling_Test.py printed).

Output: stdout and runs/ancilla_table/table.csv.
"""

import csv

import _common  # noqa: F401
import numpy as np

from colmodel import ancilla, correlations as cr, master_equation as me, unravelling as un


def row(name, rho, g=0.2):
    D, blk = me.kossakowski(rho, g, g)
    return {
        "state": name,
        "coherence": cr.local_coherence(rho),
        "negativity": cr.negativity(rho),
        "concurrence": cr.concurrence(rho),
        "discord": cr.discord(rho),
        "C(Z,Z)": cr.zz_covariance(rho),
        "class": cr.classify(rho),
        "||D_LR||": float(np.linalg.norm(blk["D_LR"])),
        "frac_L": un.monitorable_fraction(D, "L"),
    }


def main():
    tau = 1e-3
    states = [(n, ancilla.representative(n)) for n in ancilla.REPRESENTATIVES]
    states.append(("mixture", ancilla.mixture()))
    states.append((f"product, tau={tau:g}",
                   ancilla.collision_scaled_product(ancilla.representative("product"), tau)))
    states += [(f"random seed {k}", ancilla.random_state(seed=k)) for k in range(4)]

    rows = [row(n, r) for n, r in states]
    cols = list(rows[0])
    print(f"{'state':>20} {'coher.':>7} {'negat.':>7} {'concur.':>7} {'discord':>7} "
          f"{'C(Z,Z)':>7} {'class':>10} {'||D_LR||':>9} {'frac_L':>7}")
    for r in rows:
        print(f"{r['state']:>20} {r['coherence']:7.4f} {r['negativity']:7.4f} {r['concurrence']:7.4f} "
              f"{r['discord']:7.4f} {r['C(Z,Z)']:+7.4f} {r['class']:>10} {r['||D_LR||']:9.5f} {r['frac_L']:7.4f}")
    print("(||D_LR|| and frac_L with gL = gR = 0.2)")

    path = _common.out_dir("ancilla_table") / "table.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    rho = ancilla.collision_scaled_product(ancilla.representative("product"), tau)
    D, _ = me.kossakowski(rho, 1.0, 1.0)
    np.set_printoptions(precision=5, suppress=True)
    print(f"\nKossakowski matrix of the collision-scaled product (tau = {tau:g}, gL = gR = 1):")
    print(D)
    print(f"GKSL: {un.gksl_ok(D)}   strictly unravellable: {un.strict_unravellable(D)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
