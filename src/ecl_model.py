"""ECL Model Engine - IFRS 9 Expected Credit Loss Calculator
Computes Stage 1/2/3 ECL using PD, LGD, EAD components with macro overlays.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm


SCENARIO_WEIGHTS = {
    "base": 0.50,
    "adverse": 0.30,
    "severe": 0.20,
}

SCENARIO_PD_MULTIPLIERS = {
    "base": 1.0,
    "adverse": 1.8,
    "severe": 3.2,
}


def compute_lifetime_pd(pd_12m: float, lgd: float, vintage_years: int) -> float:
    """Convert 12-month PD to lifetime PD using a simple Markov chain approximation."""
    annual_survival = 1 - pd_12m
    lifetime_survival = annual_survival ** vintage_years
    return 1 - lifetime_survival


def apply_macro_overlay(pd_base: float, gdp_growth: float = 6.5, credit_growth: float = 12.0) -> float:
    """Apply macroeconomic satellite model overlay to base PD."""
    gdp_factor = max(0.5, 1 - (gdp_growth - 6.0) * 0.05)
    credit_factor = max(0.8, 1 + (credit_growth - 12.0) * 0.02)
    return pd_base * gdp_factor * credit_factor


def compute_ecl_row(row: pd.Series) -> dict:
    """Compute ECL for a single borrower row across scenarios."""
    ecl_scenarios = {}
    for scenario, weight in SCENARIO_WEIGHTS.items():
        pd_adj = row["pd"] * SCENARIO_PD_MULTIPLIERS[scenario]
        pd_adj = apply_macro_overlay(pd_adj)
        if row["ifrs9_stage"] == 1:
            ecl = pd_adj * row["lgd"] * row["ead_cr"]
        elif row["ifrs9_stage"] == 2:
            lifetime_pd = compute_lifetime_pd(pd_adj, row["lgd"], row["vintage_years"])
            ecl = lifetime_pd * row["lgd"] * row["ead_cr"]
        else:  # Stage 3 - credit impaired
            ecl = row["lgd"] * row["ead_cr"]
        ecl_scenarios[scenario] = ecl * weight
    return {
        "ecl_stage1": ecl_scenarios["base"] if row["ifrs9_stage"] == 1 else 0,
        "ecl_stage2": ecl_scenarios["base"] if row["ifrs9_stage"] == 2 else 0,
        "ecl_stage3": ecl_scenarios["base"] if row["ifrs9_stage"] == 3 else 0,
        "ecl_probability_weighted": sum(ecl_scenarios.values()),
        "ecl_coverage_ratio": sum(ecl_scenarios.values()) / row["ead_cr"] if row["ead_cr"] > 0 else 0,
    }


def run_ecl_model(df: pd.DataFrame) -> pd.DataFrame:
    """Run full ECL model on portfolio DataFrame. Returns enriched DataFrame."""
    ecl_results = df.apply(compute_ecl_row, axis=1, result_type="expand")
    df = pd.concat([df, ecl_results], axis=1)
    total_ecl = df["ecl_probability_weighted"].sum()
    total_ead = df["ead_cr"].sum()
    coverage = total_ecl / total_ead if total_ead > 0 else 0
    print(f"\n=== ECL Model Results ===")
    print(f"Total ECL (INR Cr): {total_ecl:.2f}")
    print(f"Coverage Ratio: {coverage:.2%}")
    print(f"Stage Distribution: {df['ifrs9_stage'].value_counts().to_dict()}")
    stage_ecl = {
        1: df['ecl_stage1'].sum(),
        2: df['ecl_stage2'].sum(),
        3: df['ecl_stage3'].sum()
    }
    print(f"Stage-wise ECL: {stage_ecl}")
    return df


if __name__ == "__main__":
    from data_generator import generate_portfolio
    portfolio = generate_portfolio(n_borrowers=5000)
    enriched = run_ecl_model(portfolio)
    enriched.to_csv("../data/ecl_results.csv", index=False)
    print("\nSaved to data/ecl_results.csv")
    print(enriched[["borrower_id", "sector", "ifrs9_stage", "ecl_probability_weighted", "ecl_coverage_ratio"]].head(10).to_string())
