"""Tauchen rows versus Farmer-Toda rows on the 11-point, nsds=2 grid.

Same 1985-2026Q2 cycle and same grid as explore_grid.py. Farmer-Toda is the
Newton tilt in fortran/NL.f90: start from a floored Tauchen row and choose
exponential weights so the discrete conditional mean and second moment match
N((1-rho)*mu + rho*z, sigma). If the Newton step fails, the code keeps the
floored Tauchen row.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from explore_grid import MU, RHO, SIG, farmer_toda_row, tauchen_row, z_grid

OUT = Path(__file__).resolve().parent / "explore_ft.pdf"


def rows_for_regime(z, regime):
    before = []
    after = []
    matched = []
    for z0 in z:
        mean = (1.0 - RHO[regime]) * MU[regime] + RHO[regime] * z0
        q = tauchen_row(z, mean, SIG[regime])
        p, ok = farmer_toda_row(z, q, mean, SIG[regime] ** 2)
        before.append(q)
        after.append(p)
        matched.append(ok)
    return np.vstack(before), np.vstack(after), np.array(matched)


def stay_neighbor(P, i):
    same = P[i, i]
    neighbor = 0.0
    if i > 0:
        neighbor += P[i, i - 1]
    if i + 1 < P.shape[1]:
        neighbor += P[i, i + 1]
    return same, neighbor


def draw_compare_table(pdf, z, before, after, matched):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11.0))
    fig.suptitle(
        r"Tauchen versus Farmer--Toda, 11-point grid ($n_{sd}=2$)",
        fontsize=12,
    )
    labels = [
        "i", "z",
        "P(same) T", "P(same) FT",
        "P(neigh) T", "P(neigh) FT",
        "matched",
    ]
    for s, ax in enumerate(axes):
        ax.axis("off")
        cell = []
        for i, z0 in enumerate(z):
            st, nt = stay_neighbor(before[s], i)
            sf, nf = stay_neighbor(after[s], i)
            cell.append([
                str(i),
                f"{z0:.3f}",
                f"{st:.3f}",
                f"{sf:.3f}",
                f"{nt:.3f}",
                f"{nf:.3f}",
                "yes" if matched[s][i] else "no",
            ])
        n_ok = int(matched[s].sum())
        ax.set_title(
            rf"Regime {s}: $\sigma={SIG[s]:.4f}$.  "
            rf"Moment match {n_ok}/{z.size} rows.",
            loc="left",
            fontsize=10,
        )
        table = ax.table(cellText=cell, colLabels=labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def draw_heatmaps(pdf, z, before, after):
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.5), constrained_layout=True)
    fig.suptitle(r"$P(z'\mid z,s)$: Tauchen (left) and Farmer--Toda (right)", fontsize=12)
    for s in range(3):
        for col, mat in enumerate((before[s], after[s])):
            ax = axes[s, col]
            im = ax.imshow(mat, origin="lower", vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
            ax.set_xticks(range(0, z.size, 2))
            ax.set_yticks(range(0, z.size, 2))
            ax.set_title(
                rf"regime {s}, {'Tauchen' if col == 0 else 'Farmer-Toda'}",
                fontsize=9,
            )
            if s == 2:
                ax.set_xlabel(r"$z'$ index")
            if col == 0:
                ax.set_ylabel(r"$z$ index")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="probability")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    z = z_grid()
    before, after, matched = [], [], []
    for s in range(3):
        b, a, ok = rows_for_regime(z, s)
        before.append(b)
        after.append(a)
        matched.append(ok)
        print(
            f"regime {s}: matched {int(ok.sum())}/{z.size}, "
            f"mean P(same) {b.diagonal().mean():.3f} -> {a.diagonal().mean():.3f}"
        )
    with PdfPages(OUT) as pdf:
        draw_compare_table(pdf, z, before, after, matched)
        draw_heatmaps(pdf, z, before, after)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
