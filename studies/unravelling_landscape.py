"""Single-bath unravelling of the two-bath collision model.

How much of one bath's dissipation can be monitored (peeled off as detector
clicks) while the no-click generator stays completely positive? Answered with
colmodel.unravelling.monitorable_fraction on the 4x4 Kossakowski matrix

    D = [[D_LL, D_LR], [D_LR^dag, D_RR]],   basis (s+_1, s-_1, s+_N, s-_N),

which is independent of N, J and the chain Hamiltonian: those enter only the
coherent part. Strict (full) monitoring holds iff D_LR = 0 iff fraction = 1.
Only transverse (x-y) ancilla correlations feed D_LR.

Sections (output in runs/unravelling_landscape/):
  1. named representatives       -> named_reps.csv
  2. Bell-diagonal (c1, c2) map    -> fraction_landscape.png
  3. c3 (class) invariance        -> c3_invariance.png
  4. fraction vs negativity/discord over random states -> frac_vs_correlation.png
  5. thermal marginals            -> thermal_marginals.png
  6. canonical channels (one bath vs two), printed
Printed output also goes to log.txt.
"""

import csv

import _common
import numpy as np

from colmodel import ancilla, correlations as cr, master_equation as me, unravelling as un
from colmodel.operators import is_psd, min_eig

G = 0.2                         # gL = gR
CMAP = {"classical": "tab:green", "discord": "tab:orange", "entangled": "tab:red", "product": "tab:gray"}
CLASSES = ("product", "classical", "discord", "entangled")     # fixed drawing / legend order


def D_of(rho):
    return me.kossakowski(rho, G, G)[0]


def report_state(name, rho):
    D, blk = me.kossakowski(rho, G, G)
    return {
        "name": name,
        "class": cr.classify(rho),
        "negativity": cr.negativity(rho),
        "discord": cr.discord(rho, n_grid=24),
        "||D_LR||": float(np.linalg.norm(blk["D_LR"])),
        "min_eig_D": min_eig(D),
        "GKSL": un.gksl_ok(D),
        "strict_a": un.strict_unravellable(D),
        "frac_L": un.monitorable_fraction(D, "L"),
        "frac_R": un.monitorable_fraction(D, "R"),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _common.out_dir("unravelling_landscape")
    with _common.Tee(out / "log.txt"):

        # ---- 1. named representatives ------------------------------------
        print(f"=== named representatives (gL = gR = {G}) ===")
        rows = [report_state(m, ancilla.representative(m)) for m in ancilla.REPRESENTATIVES]
        hdr = (f"{'name':>10} {'class':>10} {'neg':>7} {'disc':>7} {'||D_LR||':>9} "
               f"{'minE(D)':>9} {'GKSL':>5} {'(a)':>5} {'frac_L':>7} {'frac_R':>7}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['name']:>10} {r['class']:>10} {r['negativity']:7.3f} "
                  f"{r['discord']:7.3f} {r['||D_LR||']:9.4f} {r['min_eig_D']:9.4f} "
                  f"{str(r['GKSL']):>5} {str(r['strict_a']):>5} "
                  f"{r['frac_L']:7.3f} {r['frac_R']:7.3f}")
        with open(out / "named_reps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()

        # ---- 2. Bell-diagonal (c1, c2) landscape of frac_L -----------------
        ng = 81
        cs = np.linspace(-1, 1, ng)
        frac_grid = np.full((ng, ng), np.nan)
        neg_grid = np.full((ng, ng), np.nan)
        for i, c2 in enumerate(cs):
            for j, c1 in enumerate(cs):
                rho = ancilla.bell_diagonal(c1, c2, 0.0)
                if np.linalg.eigvalsh(rho).min() < -1e-12:      # outside the tetrahedron slice
                    continue
                frac_grid[i, j] = un.monitorable_fraction(D_of(rho), "L")
                neg_grid[i, j] = cr.negativity(rho)

        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(frac_grid, origin="lower", extent=[-1, 1, -1, 1],
                       vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.contour(cs, cs, np.nan_to_num(neg_grid, nan=-1), levels=[1e-6], colors="white", linewidths=1.5)
        ax.plot([-1, 1], [0, 0], "w:", lw=1)       # zero-discord axes (c1 = 0 or c2 = 0)
        ax.plot([0, 0], [-1, 1], "w:", lw=1)
        ax.set_xlabel("c1  (<XX>)")
        ax.set_ylabel("c2  (<YY>)")
        ax.set_title("Max monitorable fraction of LEFT bath\nBell-diagonal ancilla, c3 = 0")
        fig.colorbar(im, ax=ax, label="frac_L")
        ax.text(0.02, 0.02, "white line: separability boundary (negativity=0)\ndotted: zero-discord axes",
                transform=ax.transAxes, fontsize=7, color="w", va="bottom")
        fig.tight_layout()
        fig.savefig(out / "fraction_landscape.png", dpi=130)
        plt.close(fig)

        # ---- 3. c3 invariance: the fraction ignores the correlation class ---
        # For Bell-diagonal states D depends only on (c1, c2); c3 moves the class.
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for (c1, c2) in [(0.5, 0.0), (0.6, -0.4), (0.3, 0.3)]:
            c3s, fracs, klass = [], [], []
            for c3 in np.linspace(-1, 1, 81):
                rho = ancilla.bell_diagonal(c1, c2, c3)
                if np.linalg.eigvalsh(rho).min() < -1e-12:
                    continue
                c3s.append(c3)
                fracs.append(un.monitorable_fraction(D_of(rho), "L"))
                klass.append(cr.classify(rho, disc_grid=10))
            c3s, fracs = np.array(c3s), np.array(fracs)
            ax.plot(c3s, fracs, "-", lw=1, color="k", alpha=0.4)
            for k in [c for c in CLASSES if c in klass]:
                mask = np.array([kk == k for kk in klass])
                ax.scatter(c3s[mask], fracs[mask], s=16, color=CMAP[k], label=k, zorder=3)
            ax.annotate(f"(c1,c2)=({c1},{c2})", (c3s[-1], fracs[-1]), fontsize=7, va="center")
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=8, title="class")
        ax.set_xlabel("c3  (<ZZ>)  -- moves the correlation class")
        ax.set_ylabel("frac_L")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Monitorable fraction is FLAT in c3:\nit tracks transverse (c1,c2) correlation, not the class")
        fig.tight_layout()
        fig.savefig(out / "c3_invariance.png", dpi=130)
        plt.close(fig)

        # ---- 4. frac_L vs negativity / discord over random states ----------
        rng = np.random.default_rng(0)
        negs, discs, fr, klass = [], [], [], []
        for _ in range(120):
            rho = ancilla.random_state(seed=int(rng.integers(1 << 31)))
            D = D_of(rho)
            if not un.gksl_ok(D):
                continue
            n_i = cr.negativity(rho)
            d_i = cr.discord(rho, n_grid=10)
            negs.append(n_i)
            discs.append(d_i)
            fr.append(un.monitorable_fraction(D, "L"))
            klass.append(cr.classify(rho, discord_value=d_i))
        colors = [CMAP[k] for k in klass]

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
        axes[0].scatter(negs, fr, s=14, c=colors, alpha=0.8)
        axes[0].set_xlabel("negativity (entanglement)")
        axes[0].set_ylabel("frac_L")
        axes[0].set_title("frac_L vs negativity")
        axes[1].scatter(discs, fr, s=14, c=colors, alpha=0.8)
        axes[1].set_xlabel("discord")
        axes[1].set_ylabel("frac_L")
        axes[1].set_title("frac_L vs discord")
        for a in axes:
            a.set_ylim(-0.05, 1.05)
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=CMAP[k], label=k)
                   for k in CLASSES]
        axes[1].legend(handles=handles, fontsize=8, title="class")
        fig.suptitle("No functional link: a given class spans a range of fractions "
                     "(fraction is set by the Kossakowski data, not the class label)")
        fig.tight_layout()
        fig.savefig(out / "frac_vs_correlation.png", dpi=130)
        plt.close(fig)

        # ---- 5. thermal marginals -----------------------------------------
        print("=== thermal-marginal correlated ancillas ===")
        print("D depends only on (p_s, p_r, t1, t2); t3 (the z-z / class knob) drops out.")
        print("frac_L == frac_R = 1 - sigma_max(M)^2 exactly, even for p_s != p_r.\n")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        # (left) fixed transverse correlation, sweep t3 across classes
        p_s0, p_r0, t1_0, t2_0 = 0.45, 0.35, 0.35, 0.0
        t3s, fLs, fRs, kl = [], [], [], []
        for t3 in np.linspace(-1, 1, 121):
            rho = ancilla.thermal_correlated(p_s0, p_r0, t1_0, t2_0, t3)
            if not is_psd(rho, 1e-12):
                continue
            D = D_of(rho)
            t3s.append(t3)
            fLs.append(un.monitorable_fraction(D, "L"))
            fRs.append(un.monitorable_fraction(D, "R"))
            kl.append(cr.classify(rho, disc_grid=10))
        t3s = np.array(t3s)
        axes[0].plot(t3s, fLs, "-", lw=6, color="tab:blue", alpha=0.35, label="frac_L")
        axes[0].plot(t3s, fRs, "--", lw=1.5, color="tab:purple", label="frac_R (= frac_L)")
        for k in [c for c in CLASSES if c in kl]:
            msk = np.array([kk == k for kk in kl])
            axes[0].scatter(t3s[msk], np.full(msk.sum(), -0.03), s=12, color=CMAP[k], label=k)
        axes[0].set_xlabel("t3  (<ZZ>, correlation-class knob)")
        axes[0].set_ylabel("fraction")
        axes[0].set_ylim(-0.08, 1.05)
        axes[0].set_title(f"Thermal marginals p_s={p_s0}, p_r={p_r0}\n"
                          f"transverse (t1,t2)=({t1_0},{t2_0}): FLAT in t3, frac_L=frac_R")
        h, l = axes[0].get_legend_handles_labels()
        seen = dict(zip(l, h))
        axes[0].legend(seen.values(), seen.keys(), fontsize=7, ncol=2)

        # (right) shared fraction vs left-ancilla temperature at fixed transverse coupling
        p_r1, t1_1 = 0.30, 0.30
        ps_phys, frac_t, ps_unphys = [], [], []
        for p_s in np.linspace(0.02, 0.98, 97):
            rho = ancilla.thermal_correlated(p_s, p_r1, t1_1, 0.0, 0.0)
            if not is_psd(rho, 1e-12):
                ps_unphys.append(p_s)
                continue
            ps_phys.append(p_s)
            frac_t.append(un.monitorable_fraction(D_of(rho), "L"))
        axes[1].plot(ps_phys, frac_t, "-", color="tab:blue", lw=2, label="shared fraction (=frac_L=frac_R)")
        if ps_unphys:
            axes[1].axvspan(min(ps_unphys), max(ps_unphys), color="red", alpha=0.08)
            axes[1].text(np.mean(ps_unphys), 0.5, "unphysical\n(state not PSD)",
                         ha="center", fontsize=7, color="firebrick")
        axes[1].axvline(p_r1, color="gray", ls=":", lw=1)
        axes[1].text(p_r1, 1.0, f" p_r={p_r1}", fontsize=7, color="gray")
        axes[1].set_xlabel("p_s  (left-ancilla excited population; small = cold)")
        axes[1].set_ylabel("monitorable fraction")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title(f"Temperature sets the shared fraction\n(t1={t1_1}): further from p = 1/2 => more shared")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "thermal_marginals.png", dpi=130)
        plt.close(fig)

        # ---- 6. one bath vs two: canonical channel structure --------------
        print("=== bath structure: canonical jump channels ===")
        print(f"{'state':>22} {'shared':>7}  channels (rate | pL,pR | delocalized?)")
        print("-" * 78)
        demo = [
            ("product", ancilla.representative("product")),
            ("classical (1,0,0)", ancilla.bell_diagonal(1, 0, 0)),
            ("entangled Bell", ancilla.representative("entangled")),
            ("thermal corr (.3,.15)", ancilla.thermal_correlated(0.30, 0.15, 0.5, -0.2, 0.2)),
        ]
        for nm, rho in demo:
            chans, shared = un.canonical_channels(D_of(rho))
            desc = "  ".join(f"[{c['rate']:.3f}|{c['participation']['L']:.2f},{c['participation']['R']:.2f}|"
                             f"{'YES' if c['delocalized'] else 'no'}]" for c in chans)
            print(f"{nm:>22} {shared:7.3f}  {desc}")
        print("\n  pL,pR = left/right participation of each canonical jump operator.")
        print("  delocalized channels (both pL,pR > 0) = shared-bath (nonlocal) jumps.\n")
        print(f"figures + CSV written to: {out}")


if __name__ == "__main__":
    main()
