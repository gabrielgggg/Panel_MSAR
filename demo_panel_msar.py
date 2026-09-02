"""Simulate the 3-regime panel MS-AR DGP and recover parameters jointly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_msar import PanelMSAR, simulate_panel


def main():
    true = dict(
        g=0.018, a=1.0, rho=0.75,
        mu=(-0.20, 0.0, 0.20),
        sigma=(0.06, 0.035, 0.06),
        stay=(0.90, 0.88, 0.90),
    )
    df = simulate_panel(
        n_countries=30,
        t_min=30,
        t_max=48,
        g=true["g"], a=true["a"], rho=true["rho"],
        mu=true["mu"], sigma=true["sigma"], stay=true["stay"],
        seed=11,
    )
    print(df.head())
    print(
        f"\nN countries={df.country.nunique()}  N obs={len(df)}  "
        f"years {int(df.year.min())}-{int(df.year.max())}"
    )

    model = PanelMSAR(n_regimes=3, common_rho=True, common_sigma=False, min_t=12)
    res = model.fit(
        df["country"], df["year"], df["y"],
        n_starts=4, maxiter=250, seed=11,
        compute_se=False, store_filtered=True,
    )
    print()
    print(res.summary())
    print()
    print("True vs estimate")
    print(f"  a      {true['a']:.4f}   {res.params['a']:.4f}")
    print(f"  g      {true['g']:.4f}   {res.params['g']:.4f}")
    print(f"  rho    {true['rho']:.4f}   {res.params['rho']:.4f}")
    print(f"  mu     {true['mu']}   {res.params['mu']}")
    print(f"  sigma  {true['sigma']}   {res.params['sigma']}")
    print("  P hat:\n", res.params["P"])


if __name__ == "__main__":
    main()
