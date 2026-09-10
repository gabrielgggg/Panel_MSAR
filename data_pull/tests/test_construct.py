"""Specification tests for the shipped construction in construct.py.

These tests call the real functions (including ``build_panel``). They check
economic identities and documented rules, not a second copy of the formulas.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from construct import (  # noqa: E402
    BASE_YEAR,
    EURO_AREA_AS_OF_2015,
    EURO_FX_DONOR,
    ILO_THOUSANDS_TO_PERSONS,
    QUARTER_TO_ANNUAL,
    align_quarterly_lcu_to_annual,
    apply_imf_scale,
    base_year_fx,
    build_panel,
    choose_sa_or_nsa,
    drop_years_not_matching_annual,
    interpolate_annual_to_quarterly,
    keep_segment_containing_base_year,
    parse_quarter,
    parse_year,
    patch_implausible_quarterly_employment,
    power10_unit_factor,
    quarter_midpoint,
    rebase_quarter_to_base_usd,
    sarimax_log_seasonal_adjust,
    seasonally_adjust_nsa_countries,
    tidy_ilo_employment,
    tidy_imf_fx_annual,
    tidy_imf_gdp,
)


def _imf_q(
    country: str,
    periods: list[str],
    values: list[float],
    *,
    sa: str = "SA",
    scale: float = 0,
    derivation: str = "O",
) -> pd.DataFrame:
    n = len(periods)
    return pd.DataFrame(
        {
            "COUNTRY": [country] * n,
            "TIME_PERIOD": periods,
            "OBS_VALUE": values,
            "SCALE": [scale] * n,
            "DERIVATION_TYPE": [derivation] * n,
            "S_ADJUSTMENT": [sa] * n,
        }
    )


def _imf_a(
    country: str,
    years: list[int],
    values: list[float],
    *,
    scale: float = 0,
    derivation: str = "O",
) -> pd.DataFrame:
    n = len(years)
    return pd.DataFrame(
        {
            "COUNTRY": [country] * n,
            "TIME_PERIOD": [str(y) for y in years],
            "OBS_VALUE": values,
            "SCALE": [scale] * n,
            "DERIVATION_TYPE": [derivation] * n,
        }
    )


def _fx(country: str, year: int, rate: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COUNTRY": [country],
            "TIME_PERIOD": [str(year)],
            "OBS_VALUE": [rate],
            "SCALE": [0],
            "DERIVATION_TYPE": ["O"],
        }
    )


def _ilo(
    country: str,
    times: list[object],
    values_thousands: list[float],
    *,
    sex: str = "SEX_T",
    age: str = "AGE_YTHADULT_YGE15",
    source: str = "BA:1",
) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "ref_area": [country] * n,
            "source": [source] * n,
            "indicator": ["EMP_TEMP_SEX_AGE_NB"] * n,
            "sex": [sex] * n,
            "classif1": [age] * n,
            "time": times,
            "obs_value": values_thousands,
        }
    )


def _empty_ilo() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ref_area",
            "source",
            "indicator",
            "sex",
            "classif1",
            "time",
            "obs_value",
        ]
    )


class TestScaleAndParsing(unittest.TestCase):
    def test_scale_zero_is_identity(self):
        self.assertEqual(apply_imf_scale(4.67e12, 0), 4.67e12)

    def test_scale_six_is_millions(self):
        # SCALE is the exponent of 10; 3 with SCALE=6 is 3 million.
        self.assertEqual(apply_imf_scale(3, 6), 3_000_000)

    def test_missing_scale_treated_as_zero(self):
        self.assertEqual(apply_imf_scale(10, np.nan), 10)

    def test_parse_imf_and_ilo_quarter_labels(self):
        self.assertEqual(parse_quarter("2015-Q1"), (2015, 1))
        self.assertEqual(parse_quarter("2015Q4"), (2015, 4))
        self.assertEqual(parse_quarter("2015q2"), (2015, 2))
        self.assertIsNone(parse_quarter("2015"))

    def test_parse_anea_year_floats(self):
        # pandas reads IMF ANEA TIME_PERIOD as float64 (2015.0).
        self.assertEqual(parse_year(2015.0), 2015)
        self.assertEqual(parse_year("2015.0"), 2015)
        self.assertEqual(parse_year(2015), 2015)
        self.assertEqual(parse_year("2015"), 2015)
        self.assertIsNone(parse_year(float("nan")))
        self.assertIsNone(parse_year("nan"))


class TestRebaseIdentity(unittest.TestCase):
    def test_base_year_annualized_equals_nominal_over_fx(self):
        # If the four quarterly real flows sum to the annual real total, the
        # 2015 annualized USD level must equal 2015 current-price GDP / FX.
        real_annual = 100.0
        nom_annual = 80.0
        fx = 2.0
        real_q = real_annual / QUARTER_TO_ANNUAL
        usd = rebase_quarter_to_base_usd(real_q, real_annual, nom_annual, fx)
        self.assertEqual(usd, nom_annual / fx)

    def test_real_lcu_growth_passes_through_one_to_one(self):
        # Doubling real LCU doubles 2015-USD GDP. A quarterly FX movement
        # cannot enter: the function has no quarterly-FX argument.
        base = rebase_quarter_to_base_usd(25, 100, 80, 2)
        doubled = rebase_quarter_to_base_usd(50, 100, 80, 2)
        self.assertEqual(doubled, 2 * base)

    def test_higher_base_fx_lowers_usd_one_for_one(self):
        low_fx = rebase_quarter_to_base_usd(25, 100, 80, 2)
        high_fx = rebase_quarter_to_base_usd(25, 100, 80, 4)
        self.assertEqual(high_fx, low_fx / 2)

    def test_nonpositive_inputs_are_missing(self):
        self.assertTrue(np.isnan(rebase_quarter_to_base_usd(25, 100, 80, 0)))
        self.assertTrue(np.isnan(rebase_quarter_to_base_usd(-1, 100, 80, 2)))


class TestEuroFxFill(unittest.TestCase):
    def test_germany_without_2015_national_rate_uses_g163(self):
        er = pd.concat(
            [
                _fx("USA", 2015, 1.0),
                _fx("DEU", 1998, 1.76),  # DM era only
                _fx(EURO_FX_DONOR, 2015, 0.901),
            ],
            ignore_index=True,
        )
        tidy = tidy_imf_fx_annual(er)
        rates = base_year_fx(tidy, base_year=2015)
        self.assertIn("DEU", EURO_AREA_AS_OF_2015)
        self.assertEqual(float(rates.loc["DEU"]), 0.901)
        self.assertEqual(float(rates.loc["USA"]), 1.0)
        self.assertNotIn(EURO_FX_DONOR, rates.index)

    def test_own_2015_rate_is_not_overwritten_for_non_euro(self):
        er = pd.concat(
            [
                _fx("GBR", 2015, 0.65),
                _fx(EURO_FX_DONOR, 2015, 0.901),
            ],
            ignore_index=True,
        )
        rates = base_year_fx(tidy_imf_fx_annual(er), base_year=2015)
        self.assertEqual(float(rates.loc["GBR"]), 0.65)


class TestUnitBreaksAndEmpPatches(unittest.TestCase):
    def test_power10_ignores_factor_of_four(self):
        # SAAR vs quarterly-flow (x4) must not be treated as a unit break.
        self.assertEqual(power10_unit_factor(400.0, 100.0), 1.0)
        self.assertEqual(power10_unit_factor(100.0, 100.0), 1.0)

    def test_power10_corrects_millions_mislabel(self):
        self.assertEqual(power10_unit_factor(100.0 / 1e6, 100.0), 1e6)
        self.assertEqual(power10_unit_factor(100.0 / 1e3, 100.0), 1e3)

    def test_align_rescales_later_years_to_annual(self):
        q = pd.DataFrame(
            {
                "country": ["JAM"] * 8,
                "year": [2015] * 4 + [2018] * 4,
                "quarter": [1, 2, 3, 4] * 2,
                "period": [f"2015-Q{q}" for q in range(1, 5)]
                + [f"2018-Q{q}" for q in range(1, 5)],
                "gdp_lcu": [10.0] * 4 + [10.0 / 1e6] * 4,
                "gdp_sa": ["SA"] * 8,
            }
        )
        annual = pd.DataFrame(
            {
                "country": ["JAM", "JAM"],
                "year": [2015, 2018],
                "quarter": [pd.NA, pd.NA],
                "period": ["2015", "2018"],
                "gdp_lcu": [40.0, 40.0],
                "gdp_sa": ["A", "A"],
            }
        )
        aligned = align_quarterly_lcu_to_annual(q, annual)
        y2018 = aligned.loc[aligned.year == 2018, "gdp_lcu"]
        self.assertTrue(np.allclose(y2018, 10.0))
        y2015 = aligned.loc[aligned.year == 2015, "gdp_lcu"]
        self.assertTrue(np.allclose(y2015, 10.0))

    def test_drop_years_whose_quarterly_still_mismatches_annual(self):
        q = pd.DataFrame(
            {
                "country": ["SLV"] * 8,
                "year": [2004] * 4 + [2005] * 4,
                "quarter": [1, 2, 3, 4] * 2,
                "period": [f"2004-Q{q}" for q in range(1, 5)]
                + [f"2005-Q{q}" for q in range(1, 5)],
                "gdp_lcu": [10.0 / 50.0] * 4 + [10.0] * 4,
                "gdp_sa": ["SA"] * 8,
                "gdp_unit_factor": [1.0] * 8,
            }
        )
        annual = pd.DataFrame(
            {
                "country": ["SLV", "SLV"],
                "year": [2004, 2005],
                "gdp_lcu": [40.0, 40.0],
            }
        )
        kept = drop_years_not_matching_annual(q, annual)
        self.assertEqual(set(kept.year), {2005})

    def test_keep_only_spell_that_includes_base_year(self):
        q = pd.DataFrame(
            {
                "country": ["SLV"] * 8,
                "year": [2004] * 4 + [2015] * 4,
                "quarter": [1, 2, 3, 4] * 2,
                "period": [f"2004-Q{q}" for q in range(1, 5)]
                + [f"2015-Q{q}" for q in range(1, 5)],
                "gdp_lcu": [10.0] * 4 + [500.0] * 4,
                "gdp_sa": ["SA"] * 8,
            }
        )
        kept = keep_segment_containing_base_year(q, base_year=2015, jump=20.0)
        self.assertEqual(set(kept.year), {2015})
        self.assertTrue((kept.gdp_lcu == 500.0).all())

    def test_quarterly_employment_far_below_annual_is_replaced(self):
        q = pd.DataFrame(
            {
                "country": ["ISR"] * 4,
                "year": [2017] * 4,
                "quarter": [1, 2, 3, 4],
                "period": [f"2017-Q{q}" for q in range(1, 5)],
                "emp_persons": [900_000.0] * 4,
                "emp_source": ["official-q"] * 4,
                "emp_age": ["AGE_YTHADULT_YGE15"] * 4,
                "emp_ilo_source": ["BA:1"] * 4,
            }
        )
        a = pd.DataFrame(
            {
                "country": ["ISR", "ISR"],
                "year": [2016, 2018],
                "quarter": [pd.NA, pd.NA],
                "period": ["2016", "2018"],
                "emp_persons": [3_600_000.0, 3_800_000.0],
                "emp_source": ["official-a"] * 2,
                "emp_age": ["AGE_YTHADULT_YGE15"] * 2,
                "emp_ilo_source": ["BA:1"] * 2,
            }
        )
        patched = patch_implausible_quarterly_employment(q, a)
        # Interpolated annual at 2017 mid-year is 3.7m; 0.9m / 3.7m < 0.4.
        self.assertTrue((patched.emp_persons > 3_000_000).all())
        self.assertTrue(patched.emp_source.str.contains("replaced").all())


class TestSarimaxSeasonalAdjust(unittest.TestCase):
    def test_quarterly_means_flatten_on_known_seasonal_pattern(self):
        # 8 years of a trend times a fixed quarterly pattern. After SARIMAX
        # dummy adjustment, the range of quarter-of-year means of log y
        # must shrink relative to the raw series (the pattern is identified
        # from calendar quarter, not position).
        n = 32
        quarters = np.array([1, 2, 3, 4] * 8)
        t = np.arange(n, dtype=float)
        seas = np.array([1.20, 0.88, 0.85, 1.07])
        y = 50.0 * (1.01 ** t) * seas[quarters - 1]
        sa, note = sarimax_log_seasonal_adjust(y, quarters, min_t=12)
        self.assertIsNotNone(sa)
        self.assertIn("SARIMAX", note)

        def q_mean_range(vals: np.ndarray) -> float:
            logs = np.log(vals)
            means = [float(np.mean(logs[quarters == q])) for q in (1, 2, 3, 4)]
            return max(means) - min(means)

        self.assertLess(q_mean_range(sa), 0.5 * q_mean_range(y))

    def test_too_short_series_is_not_adjusted(self):
        y = np.array([1.0, 1.1, 0.9, 1.05, 1.02, 1.12, 0.95, 1.08])
        q = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        sa, note = sarimax_log_seasonal_adjust(y, q, min_t=12)
        self.assertIsNone(sa)
        self.assertIn("too few", note)

    def test_nsa_countries_adjusted_sa_countries_untouched(self):
        n = 24
        quarters = [1, 2, 3, 4] * 6
        years = [2015 + i // 4 for i in range(n)]
        periods = [f"{y}-Q{q}" for y, q in zip(years, quarters)]
        t = np.arange(n, dtype=float)
        seas = np.array([1.15, 0.90, 0.88, 1.07])
        nsa_y = 40.0 * (1.008 ** t) * seas[np.array(quarters) - 1]
        sa_y = 40.0 * (1.008 ** t)  # already flat seasonals
        panel = pd.DataFrame(
            {
                "country": ["NSA"] * n + ["SA0"] * n,
                "period": periods * 2,
                "year": years * 2,
                "quarter": quarters * 2,
                "gdp_per_worker": np.concatenate([nsa_y, sa_y]),
                "emp_persons": [1000.0] * (2 * n),
                "gdp_real_2015usd": np.concatenate([nsa_y, sa_y]) * 1000.0,
                "gdp_sa": ["NSA"] * n + ["SA"] * n,
                "metadata": ["orig"] * (2 * n),
            }
        )
        out = seasonally_adjust_nsa_countries(panel, min_t=12)
        sa0 = out.loc[out.country == "SA0", "gdp_per_worker"].to_numpy()
        self.assertTrue(np.allclose(sa0, sa_y))
        self.assertTrue((out.loc[out.country == "SA0", "gdp_sa"] == "SA").all())
        nsa = out.loc[out.country == "NSA"]
        self.assertTrue((nsa.gdp_sa == "SARIMAX").all())
        self.assertTrue(nsa.metadata.str.contains("SARIMAX").all())
        # Identity gdp_real = gdp_per_worker * emp is restored after SA.
        self.assertTrue(
            np.allclose(nsa.gdp_real_2015usd, nsa.gdp_per_worker * nsa.emp_persons)
        )


class TestSeasonalChoice(unittest.TestCase):
    def test_sa_country_does_not_mix_in_nsa(self):
        sa = tidy_imf_gdp(
            _imf_q("AAA", ["2015-Q1", "2015-Q2"], [1.0, 2.0], sa="SA"),
            frequency="Q",
            adjustment="SA",
        )
        nsa = tidy_imf_gdp(
            pd.concat(
                [
                    _imf_q("AAA", ["2015-Q1", "2015-Q2"], [9.0, 9.0], sa="NSA"),
                    _imf_q("BBB", ["2015-Q1", "2015-Q2"], [3.0, 4.0], sa="NSA"),
                ],
                ignore_index=True,
            ),
            frequency="Q",
            adjustment="NSA",
        )
        chosen = choose_sa_or_nsa(sa, nsa)
        aaa = chosen.loc[chosen.country == "AAA"]
        self.assertTrue((aaa.gdp_sa == "SA").all())
        self.assertEqual(set(aaa.gdp_lcu), {1.0, 2.0})
        bbb = chosen.loc[chosen.country == "BBB"]
        self.assertTrue((bbb.gdp_sa == "NSA").all())


class TestEmploymentInterpolation(unittest.TestCase):
    def test_midyear_recovers_the_annual_observation(self):
        annual = pd.DataFrame(
            {
                "country": ["ZZZ", "ZZZ"],
                "year": [2000, 2001],
                "quarter": [pd.NA, pd.NA],
                "period": ["2000", "2001"],
                "emp_persons": [100.0, 120.0],
                "emp_source": ["annual", "annual"],
                "emp_age": ["AGE_YTHADULT_YGE15", "AGE_YTHADULT_YGE15"],
                "emp_ilo_source": ["BA:1", "BA:1"],
            }
        )
        q = interpolate_annual_to_quarterly(annual)
        # Q2 midpoint is year+0.375, Q3 is year+0.625; neither is exactly
        # mid-year. The interpolant at t=year+0.5 must equal the annual
        # value — check via the same midpoints bracketing 2000.5.
        t = quarter_midpoint(2000, 3)  # 2000.625
        # Between 2000.5 (100) and 2001.5 (120), slope = 20 per year.
        expected = 100.0 + 20.0 * (t - 2000.5)
        got = float(q.loc[q.period == "2000-Q3", "emp_persons"].iloc[0])
        self.assertAlmostEqual(got, expected)

    def test_no_extrapolation_before_first_or_after_last_annual(self):
        annual = pd.DataFrame(
            {
                "country": ["ZZZ", "ZZZ"],
                "year": [2000, 2001],
                "quarter": [pd.NA, pd.NA],
                "period": ["2000", "2001"],
                "emp_persons": [100.0, 120.0],
                "emp_source": ["annual", "annual"],
                "emp_age": ["AGE_YTHADULT_YGE15", "AGE_YTHADULT_YGE15"],
                "emp_ilo_source": ["BA:1", "BA:1"],
            }
        )
        q = interpolate_annual_to_quarterly(annual)
        # First annual at 2000.5 → Q1 2000 (2000.125) is outside and dropped.
        self.assertNotIn("2000-Q1", set(q.period))
        # Last annual at 2001.5 → Q3 2001 (2001.625) is outside and dropped;
        # Q2 2001 (2001.375) is inside.
        self.assertIn("2001-Q2", set(q.period))
        self.assertNotIn("2001-Q3", set(q.period))
        self.assertNotIn("1999-Q4", set(q.period))


class TestIloFilters(unittest.TestCase):
    def test_drops_male_only_and_converts_thousands_to_persons(self):
        raw = pd.concat(
            [
                _ilo("USA", ["2015Q1"], [150.0], sex="SEX_M"),
                _ilo("USA", ["2015Q1"], [160.0], sex="SEX_T"),
            ],
            ignore_index=True,
        )
        tidy = tidy_ilo_employment(
            raw, frequency="Q", series_label="official-q"
        )
        self.assertEqual(len(tidy), 1)
        self.assertEqual(float(tidy.emp_persons.iloc[0]), 160.0 * ILO_THOUSANDS_TO_PERSONS)


def _minimal_raw_for_panel() -> dict:
    """A two-country, two-quarter toy extract that exercises build_panel."""
    # Eight quarters so the country clears MIN_OFFICIAL_Q (=8) for official
    # quarterly employment. 2016 is used only for that count; the 2015 USD
    # identity is checked on the 2015 rows.
    periods = [
        f"{y}-Q{q}" for y in (BASE_YEAR, BASE_YEAR + 1) for q in (1, 2, 3, 4)
    ]
    # AAA: real quarterly flows 10 each → 2015 annual real 40; nominal annual 40;
    # FX=2 → annualized USD = (10*4/40)*(40/2)=20 per quarter.
    # Employment 1 thousand → 1000 persons. GDP per worker = 20/1000=0.02.
    sa = pd.concat(
        [
            _imf_q("AAA", periods, [10] * 8),
            _imf_q("DEU", periods, [10] * 8),
        ],
        ignore_index=True,
    )
    nsa = _imf_q("AAA", periods, [99] * 8, sa="NSA")
    anea_real = pd.concat(
        [_imf_a("AAA", [BASE_YEAR], [40]), _imf_a("DEU", [BASE_YEAR], [40])],
        ignore_index=True,
    )
    anea_nom = pd.concat(
        [_imf_a("AAA", [BASE_YEAR], [40]), _imf_a("DEU", [BASE_YEAR], [40])],
        ignore_index=True,
    )
    er = pd.concat(
        [
            _fx("AAA", BASE_YEAR, 2.0),
            _fx(EURO_FX_DONOR, BASE_YEAR, 2.0),
            # DEU has no 2015 national FX on purpose.
        ],
        ignore_index=True,
    )
    times_q = [f"{y}Q{q}" for y in (BASE_YEAR, BASE_YEAR + 1) for q in (1, 2, 3, 4)]
    ilo_q = pd.concat(
        [
            _ilo("AAA", times_q, [1] * 8),
            _ilo("DEU", times_q, [1] * 8),
        ],
        ignore_index=True,
    )
    names = pd.DataFrame(
        {"iso3": ["AAA", "DEU"], "country_name": ["Toy", "Germany"]}
    )
    return {
        "qnea_sa": sa,
        "qnea_nsa": nsa,
        "anea_real": anea_real,
        "anea_nom": anea_nom,
        "er_annual": er,
        "ilo_official_q": ilo_q,
        "ilo_official_a": _empty_ilo(),
        "ilo_modelled_a": _empty_ilo(),
        "country_names": names,
    }


class TestBuildPanelEntryPoint(unittest.TestCase):
    def test_build_panel_identities_and_metadata(self):
        raw = _minimal_raw_for_panel()
        panel = build_panel(**raw)
        self.assertGreaterEqual(len(panel), 8)
        self.assertFalse(panel.duplicated(["country", "period"]).any())
        aaa = panel.loc[
            (panel.country == "AAA") & (panel.year == BASE_YEAR)
        ]
        # Identity: annualized GDP USD = 20, persons = 1000, per worker = 0.02.
        self.assertTrue(np.allclose(aaa.gdp_real_2015usd, 20.0))
        self.assertTrue(np.allclose(aaa.emp_persons, 1000.0))
        self.assertTrue(np.allclose(aaa.gdp_per_worker, 0.02))
        self.assertTrue((aaa.gdp_sa == "SA").all())
        meta = str(aaa.metadata.iloc[0])
        self.assertIn("IMF QNEA", meta)
        self.assertIn("IMF ER", meta)
        self.assertIn("ILOSTAT", meta)
        self.assertIn("not PPP", meta)
        # Required delivered columns exist.
        for col in ("country", "period", "gdp_per_worker", "metadata"):
            self.assertIn(col, panel.columns)
        # Germany inherited euro FX; metadata must say so.
        deu_meta = str(panel.loc[panel.country == "DEU", "metadata"].iloc[0])
        self.assertIn(EURO_FX_DONOR, deu_meta)

    def test_staff_projections_are_dropped(self):
        raw = _minimal_raw_for_panel()
        extra = _imf_q(
            "AAA",
            [f"{BASE_YEAR + 2}-Q1"],
            [10.0],
            derivation="SP",
        )
        raw["qnea_sa"] = pd.concat([raw["qnea_sa"], extra], ignore_index=True)
        panel = build_panel(**raw)
        self.assertNotIn(
            f"{BASE_YEAR + 2}-Q1",
            set(panel.loc[panel.country == "AAA", "period"]),
        )

    def test_annual_employment_is_used_when_quarterly_is_absent(self):
        raw = _minimal_raw_for_panel()
        raw["ilo_official_q"] = _empty_ilo()
        # Two annual points so interpolation has an interior interval.
        raw["ilo_official_a"] = pd.concat(
            [
                _ilo("AAA", [BASE_YEAR, BASE_YEAR + 1], [1.0, 1.0]),
                _ilo("DEU", [BASE_YEAR, BASE_YEAR + 1], [1.0, 1.0]),
            ],
            ignore_index=True,
        )
        panel = build_panel(**raw)
        self.assertGreater(len(panel), 0)
        self.assertTrue(
            panel.emp_source.str.contains("interpolat", case=False).all()
        )
        self.assertTrue(
            panel.emp_source.str.contains("annual", case=False).all()
        )

    def test_sarimax_nsa_flag_default_off(self):
        raw = _raw_with_nsa_country()
        off = build_panel(**raw)
        on = build_panel(**raw, sarimax_nsa=True)
        nsa_off = off.loc[off.country == "CCC", "gdp_sa"]
        nsa_on = on.loc[on.country == "CCC", "gdp_sa"]
        self.assertGreater(len(nsa_off), 0)
        self.assertTrue((nsa_off == "NSA").all())
        self.assertTrue((nsa_on == "SARIMAX").all())
        # IMF-SA countries are never sent through SARIMAX.
        self.assertTrue((off.loc[off.country == "AAA", "gdp_sa"] == "SA").all())
        self.assertTrue((on.loc[on.country == "AAA", "gdp_sa"] == "SA").all())


def _raw_with_nsa_country() -> dict:
    """Minimal panel plus an NSA-only country with 16 quarters (SARIMAX-eligible)."""
    raw = _minimal_raw_for_panel()
    years = list(range(BASE_YEAR, BASE_YEAR + 4))
    periods = [f"{y}-Q{q}" for y in years for q in (1, 2, 3, 4)]
    times_q = [f"{y}Q{q}" for y in years for q in (1, 2, 3, 4)]
    # Mild seasonal pattern in LCU flows so the SARIMAX dummies have signal.
    seas = [11.0, 9.0, 8.5, 11.5]
    vals = (seas * 4)
    raw["qnea_nsa"] = pd.concat(
        [raw["qnea_nsa"], _imf_q("CCC", periods, vals, sa="NSA")],
        ignore_index=True,
    )
    raw["anea_real"] = pd.concat(
        [raw["anea_real"], _imf_a("CCC", [BASE_YEAR], [sum(seas)])],
        ignore_index=True,
    )
    raw["anea_nom"] = pd.concat(
        [raw["anea_nom"], _imf_a("CCC", [BASE_YEAR], [sum(seas)])],
        ignore_index=True,
    )
    raw["er_annual"] = pd.concat(
        [raw["er_annual"], _fx("CCC", BASE_YEAR, 2.0)],
        ignore_index=True,
    )
    raw["ilo_official_q"] = pd.concat(
        [raw["ilo_official_q"], _ilo("CCC", times_q, [1] * 16)],
        ignore_index=True,
    )
    raw["country_names"] = pd.concat(
        [raw["country_names"], pd.DataFrame({"iso3": ["CCC"], "country_name": ["NsaToy"]})],
        ignore_index=True,
    )
    return raw


if __name__ == "__main__":
    unittest.main()
