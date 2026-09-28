"""One-step movement on an 11-point z grid (nsds=2) for the 1985-2026Q2 cycle.

Grid matches discretizeMSAR in fortran/NL.f90:
  half = nsds * max_s sigma_s / sqrt(1-rho_s^2)
  z from min(mu)-half to max(mu)+half, nn equally spaced points.

P(z'|z,s) is the Tauchen assignment of the conditional normal under the
regime s that produces z',
  N((1-rho_s)*mu_s + rho_s*z, sigma_s)
onto bins with edges halfway between nodes (open at the ends).
NL.f90 then adjusts each row with Farmer-Toda so the discrete row matches
that conditional mean and variance. These exhibits are the Tauchen rows,
before that adjustment.

Regimes are ordered by increasing mu.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

# 1985Q1-2026Q2, within-country g, common a, E[z]=0. Ex US, Germany, Japan.
MU = np.array([-0.7737, -0.7471, 1.1467])
RHO = np.array([0.9900, 0.9900, 0.9900])
SIG = np.array([0.0733, 0.0159, 0.0098])
NN = 11
NSDS = 2.0

OUT = Path(__file__).resolve().parent / "explore_grid.pdf"


def z_grid():
    uncond_sd = SIG / np.sqrt(1.0 - RHO ** 2)
    half = NSDS * float(np.max(uncond_sd))
    return np.linspace(float(np.min(MU)) - half, float(np.max(MU)) + half, NN)


def tauchen_row(z, mean, sd):
    """Probability of each node. Same bin edges as tauchenTransitions."""
    nn = z.size
    p = np.zeros(nn)
    if nn == 1:
        p[0] = 1.0
        return p
    for j in range(nn):
        if j == 0:
            edge = 0.5 * (z[0] + z[1])
            p[j] = norm.cdf((edge - mean) / sd)
        elif j == nn - 1:
            edge = 0.5 * (z[-1] + z[-2])
            p[j] = norm.sf((edge - mean) / sd)
        else:
            left = 0.5 * (z[j] + z[j - 1])
            right = 0.5 * (z[j + 1] + z[j])
            p[j] = norm.cdf((right - mean) / sd) - norm.cdf((left - mean) / sd)
    p = np.maximum(p, 0.0)
    total = p.sum()
    if total <= 0.0:
        raise RuntimeError("Tauchen row sums to 0")
    return p / total


def farmer_toda_row(z, qrow, mean, var):
    """Match farmerTodaRow in fortran/NL.f90. Returns (p, matched)."""
    nn = z.size
    qsum = float(np.sum(qrow))
    if nn == 1 or qsum <= 0.0:
        p = np.zeros(nn)
        p[0] = 1.0
        return p, False
    qq = np.maximum(qrow, 0.0)
    qq = (1.0 - 1.0e-12) * qq / qq.sum() + 1.0e-12 / nn
    t2bar = mean * mean + max(var, 0.0)
    z1 = z - mean
    z2 = z * z - t2bar
    lam1 = 0.0
    lam2 = 0.0
    for _ in range(80):
        expo = np.clip(lam1 * z1 + lam2 * z2, -60.0, 60.0)
        ww = qq * np.exp(expo)
        sw = float(ww.sum())
        if not (sw > 0.0) or not np.isfinite(sw):
            break
        prow = ww / sw
        g1 = float(np.sum(prow * z1))
        g2 = float(np.sum(prow * z2))
        if max(abs(g1), abs(g2)) < 1.0e-10:
            return prow, True
        h11 = float(np.sum(prow * z1 * z1) - g1 * g1) + 1.0e-10
        h12 = float(np.sum(prow * z1 * z2) - g1 * g2)
        h22 = float(np.sum(prow * z2 * z2) - g2 * g2) + 1.0e-10
        det = h11 * h22 - h12 * h12
        if abs(det) < 1.0e-18:
            break
        d1 = (h22 * g1 - h12 * g2) / det
        d2 = (-h12 * g1 + h11 * g2) / det
        lam1 -= d1
        lam2 -= d2
    return qq, False


def transition(z, regime):
    rows = []
    for z0 in z:
        mean = (1.0 - RHO[regime]) * MU[regime] + RHO[regime] * z0
        rows.append(tauchen_row(z, mean, SIG[regime]))
    return np.vstack(rows)


def movement_table(z, regime, P):
    step = z[1] - z[0]
    lines = []
    for i, z0 in enumerate(z):
        drift = (1.0 - RHO[regime]) * (MU[regime] - z0)
        same = P[i, i]
        neighbor = 0.0
        if i > 0:
            neighbor += P[i, i - 1]
        if i < z.size - 1:
            neighbor += P[i, i + 1]
        lines.append((i, z0, drift, drift / step, same, neighbor, 1.0 - same - neighbor))
    return lines


def draw_tables(pdf, z, mats):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11.0))
    fig.suptitle(
        r"One-step movement on the 11-point grid ($n_{sd}=2$)",
        fontsize=12,
    )
    col_labels = ["i", "z", "drift", "drift/step", "P(same)", "P(neighbor)", "P(further)"]
    for s, ax in enumerate(axes):
        ax.axis("off")
        rows = movement_table(z, s, mats[s])
        cell = []
        for i, z0, drift, dstep, same, neigh, further in rows:
            cell.append([
                str(i),
                f"{z0:.3f}",
                f"{drift:.4f}",
                f"{dstep:.3f}",
                f"{same:.3f}",
                f"{neigh:.3f}",
                f"{further:.3f}",
            ])
        ax.set_title(
            rf"Regime {s}:  $\mu={MU[s]:.4f}$,  $\rho={RHO[s]:.4f}$,  $\sigma={SIG[s]:.4f}$",
            loc="left",
            fontsize=10,
        )
        table = ax.table(cellText=cell, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def draw_heatmaps(pdf, z, mats):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharey=True, constrained_layout=True)
    fig.suptitle(r"$P(z'\mid z,s)$ before Farmer--Toda, 11-point grid", fontsize=12)
    for s, ax in enumerate(axes):
        im = ax.imshow(mats[s], origin="lower", vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        ax.set_xticks(range(z.size))
        ax.set_yticks(range(z.size))
        ax.set_xticklabels([f"{v:.2f}" for v in z], rotation=90, fontsize=7)
        ax.set_yticklabels([f"{v:.2f}" for v in z], fontsize=7)
        ax.set_xlabel(r"$z'$")
        if s == 0:
            ax.set_ylabel(r"$z$")
        ax.set_title(rf"regime {s}, $\sigma={SIG[s]:.3f}$", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="probability")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    z = z_grid()
    mats = [transition(z, s) for s in range(3)]
    print("z grid:")
    print(np.array2string(z, precision=4))
    print(f"step = {z[1] - z[0]:.4f}")
    with PdfPages(OUT) as pdf:
        draw_tables(pdf, z, mats)
        draw_heatmaps(pdf, z, mats)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
