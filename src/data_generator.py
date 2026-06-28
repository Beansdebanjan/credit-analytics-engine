"""
data_generator.py
Generates synthetic ABL credit portfolio data for ECL modelling.
Portfolio: INR 2,400 Cr | 500 borrowers | Basel III + IFRS 9 aligned
Author: Beansdebanjan
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

SECTORS = ["Manufacturing", "Real Estate", "Retail", "Infrastructure", "Agri", "MSME", "Healthcare"]
COLLATERAL_TYPES = ["Property", "Equipment", "Inventory", "Receivables", "Mixed"]
GEOGRAPHIES = ["Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", "Delhi", "Rajasthan"]


def generate_portfolio(n_borrowers: int = 500, total_portfolio_cr: float = 2400.0) -> pd.DataFrame:
    """
    Generate synthetic ABL loan portfolio.
    
    Args:
        n_borrowers: Number of borrowers
        total_portfolio_cr: Total portfolio size in INR Crores
    
    Returns:
        DataFrame with borrower-level credit data
    """
    data = []
    exposure_weights = np.random.dirichlet(np.ones(n_borrowers)) * total_portfolio_cr

    for i in range(n_borrowers):
        sector = random.choice(SECTORS)
        vintage = random.randint(1, 8)  # years since loan origination
        bureau_score = int(np.random.normal(680, 80))
        bureau_score = max(300, min(900, bureau_score))

        # PD based on bureau score band
        if bureau_score >= 750:
            pd_base = np.random.uniform(0.005, 0.02)
        elif bureau_score >= 650:
            pd_base = np.random.uniform(0.02, 0.08)
        else:
            pd_base = np.random.uniform(0.08, 0.25)

        # LGD based on collateral type
        collateral = random.choice(COLLATERAL_TYPES)
        lgd_map = {"Property": 0.25, "Equipment": 0.40, "Inventory": 0.55, "Receivables": 0.45, "Mixed": 0.35}
        lgd = lgd_map[collateral] + np.random.normal(0, 0.05)
        lgd = max(0.10, min(0.90, lgd))

        ead = exposure_weights[i]
        ecl_12m = pd_base * lgd * ead

        # IFRS 9 staging
        if pd_base < 0.02:
            stage = 1
            ecl = ecl_12m
        elif pd_base < 0.10:
            stage = 2
            ecl = pd_base * lgd * ead * vintage  # lifetime
        else:
            stage = 3
            ecl = lgd * ead  # credit-impaired

        data.append({
            "borrower_id": f"BRW_{i+1:04d}",
            "sector": sector,
            "geography": random.choice(GEOGRAPHIES),
            "collateral_type": collateral,
            "vintage_years": vintage,
            "bureau_score": bureau_score,
            "ead_cr": round(ead, 2),
            "pd": round(pd_base, 4),
            "lgd": round(lgd, 4),
            "ecl_cr": round(ecl, 4),
            "ifrs9_stage": stage,
            "origination_date": (datetime.today() - timedelta(days=vintage * 365)).strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(data)
    print(f"Portfolio generated: {n_borrowers} borrowers | Total EAD: INR {df['ead_cr'].sum():.1f} Cr")
    print(f"Stage distribution: {df['ifrs9_stage'].value_counts().to_dict()}")
    print(f"Total ECL: INR {df['ecl_cr'].sum():.2f} Cr")
    return df


if __name__ == "__main__":
    df = generate_portfolio()
    df.to_csv("data/portfolio.csv", index=False)
    print("\nSaved to data/portfolio.csv")
    print(df.head(10).to_string())
