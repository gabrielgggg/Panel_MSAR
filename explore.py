"""Unconditional and one-step conditional densities for the EMBI 1990 MS-AR.

Regimes are ordered by mu. The unconditional law of an AR(1) that stays in
regime s is N(mu_s, sigma_s / sqrt(1-rho_s^2)). The conditional law of the
next z, given current z and current regime s, is
N((1-rho_s)*mu_s + rho_s*z, sigma_s). Both match panel_msar.simulate_panel.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# EMBI countries, 1990+, common growth trend removed, Barro catch-up, E[z]=0.
MU = np.array([-0.0845, -0.0161, 0.1099])
RHO = np.array([0.9900, 0.7826, 0.9900])
SIG = np.array([0.0243, 0.0892, 0.0095])

OUT = Path(__file__).resolve().parent / "regime_ar_densities.pdf"


def normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def main():
    uncond_sd = SIG / np.sqrt(1.0 - RHO ** 2)
    lo = MU[0] - 2.0 * uncond_sd[0]
    hi = MU[2] + 2.0 * uncond_sd[2]
    grid = np.linspace(lo, hi, 11)
    # Third grid point from the left.
    z_now = grid[2]
    cond_mean = (1.0 - RHO[1]) * MU[1] + RHO[1] * z_now

    xs = np.linspace(lo - 0.15, hi + 0.55, 1200)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#1f4e79", "#c45911", "#548235"]
    labels = [
        r"uncond. regime 0",
        r"uncond. regime 1 (middle)",
        r"uncond. regime 2",
    ]
    for i in range(3):
        ax.plot(xs, normal_pdf(xs, MU[i], uncond_sd[i]), color=colors[i], lw=1.8, label=labels[i])
    ax.plot(
        xs,
        normal_pdf(xs, cond_mean, SIG[1]),
        color="#7030a0",
        lw=1.8,
        ls="--",
        label=r"cond. at 3rd grid point, regime 1",
    )
    ax.scatter(grid, np.zeros_like(grid), s=28, c="0.15", zorder=5, clip_on=False)
    ax.scatter([z_now], [0.0], s=54, c="#7030a0", zorder=6, clip_on=False)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel("density")
    ax.set_title(r"Within-regime AR(1) densities, EMBI 1990 catch-up cycle")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"uncond sd {uncond_sd}")
    print(f"grid {grid}")
    print(f"3rd grid point {z_now}, cond mean {cond_mean}, cond sd {SIG[1]}")


if __name__ == "__main__":
    main()
