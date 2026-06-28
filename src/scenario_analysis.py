import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StressScenario:
    """Defines a stress test scenario with macroeconomic shock parameters."""
    name: str
    description: str
    rate_shock_bps: float          # Interest rate shock in basis points
    gdp_contraction_pct: float     # GDP contraction percentage
    default_rate_multiplier: float # Multiplier on base default rates
    lgd_adjustment: float          # Loss Given Default adjustment
    sector_shocks: Dict[str, float] = field(default_factory=dict)  # Sector-specific spread shocks


# Standard Basel III / EBA stress scenarios
PREDEFINED_SCENARIOS = [
    StressScenario(
        name='Mild Recession',
        description='Moderate economic slowdown with gradual rate increases',
        rate_shock_bps=150,
        gdp_contraction_pct=-1.5,
        default_rate_multiplier=1.5,
        lgd_adjustment=0.05,
        sector_shocks={'Real Estate': 0.03, 'Retail': 0.02, 'Energy': 0.02}
    ),
    StressScenario(
        name='Severe Recession',
        description='Deep recession with significant credit deterioration',
        rate_shock_bps=300,
        gdp_contraction_pct=-4.0,
        default_rate_multiplier=3.0,
        lgd_adjustment=0.15,
        sector_shocks={'Real Estate': 0.08, 'Retail': 0.06, 'Energy': 0.07,
                       'Financials': 0.05, 'Consumer Discretionary': 0.06}
    ),
    StressScenario(
        name='Financial Crisis',
        description='Systemic financial crisis with liquidity freeze (2008-style)',
        rate_shock_bps=500,
        gdp_contraction_pct=-6.5,
        default_rate_multiplier=5.0,
        lgd_adjustment=0.25,
        sector_shocks={'Financials': 0.15, 'Real Estate': 0.12, 'Retail': 0.10,
                       'Energy': 0.08, 'Industrials': 0.07, 'Consumer Discretionary': 0.09}
    ),
    StressScenario(
        name='Rate Shock',
        description='Rapid interest rate increase causing repricing risk',
        rate_shock_bps=400,
        gdp_contraction_pct=-0.5,
        default_rate_multiplier=1.8,
        lgd_adjustment=0.08,
        sector_shocks={'Real Estate': 0.10, 'Utilities': 0.06, 'Financials': 0.04}
    ),
    StressScenario(
        name='Sector Liquidity Crunch',
        description='Sector-specific liquidity event affecting concentrated exposures',
        rate_shock_bps=200,
        gdp_contraction_pct=-2.0,
        default_rate_multiplier=2.5,
        lgd_adjustment=0.12,
        sector_shocks={'Energy': 0.15, 'Commodities': 0.12, 'Mining': 0.10}
    )
]


class ScenarioAnalyzer:
    """
    Credit Portfolio Scenario Analysis and Stress Testing Framework.
    Implements forward-looking stress scenarios based on Basel III / EBA ICAAP guidelines.
    """

    def __init__(self, portfolio_df: pd.DataFrame):
        """
        Initialize with portfolio DataFrame.
        Expected columns: company_id, company_name, sector, exposure_usd,
                          pd (probability of default), lgd, rating
        """
        self.portfolio = portfolio_df.copy()
        self.total_exposure = portfolio_df['exposure_usd'].sum()
        self.base_expected_loss = self._compute_base_el()

    def _compute_base_el(self) -> float:
        """Compute baseline Expected Loss (EL = EAD * PD * LGD)."""
        if 'pd' in self.portfolio.columns and 'lgd' in self.portfolio.columns:
            el = (self.portfolio['exposure_usd'] *
                  self.portfolio['pd'] *
                  self.portfolio['lgd']).sum()
            return float(el)
        return 0.0

    def apply_scenario(self, scenario: StressScenario) -> pd.DataFrame:
        """
        Apply a stress scenario to the portfolio and compute stressed metrics.
        Returns portfolio DataFrame with stressed PD, LGD and Expected Loss.
        """
        stressed = self.portfolio.copy()

        if 'pd' in stressed.columns:
            stressed['stressed_pd'] = np.clip(
                stressed['pd'] * scenario.default_rate_multiplier, 0, 1
            )
        else:
            stressed['stressed_pd'] = np.clip(
                0.02 * scenario.default_rate_multiplier, 0, 1
            )

        if 'lgd' in stressed.columns:
            stressed['stressed_lgd'] = np.clip(
                stressed['lgd'] + scenario.lgd_adjustment, 0, 1
            )
        else:
            stressed['stressed_lgd'] = np.clip(0.45 + scenario.lgd_adjustment, 0, 1)

        # Apply sector-specific shocks
        if scenario.sector_shocks and 'sector' in stressed.columns:
            for sector, shock in scenario.sector_shocks.items():
                mask = stressed['sector'] == sector
                stressed.loc[mask, 'stressed_lgd'] = np.clip(
                    stressed.loc[mask, 'stressed_lgd'] + shock, 0, 1
                )

        stressed['stressed_el'] = (
            stressed['exposure_usd'] *
            stressed['stressed_pd'] *
            stressed['stressed_lgd']
        )

        stressed['el_increase'] = stressed['stressed_el'] - (
            stressed.get('base_el',
                stressed['exposure_usd'] *
                stressed.get('pd', pd.Series([0.02] * len(stressed))) *
                stressed.get('lgd', pd.Series([0.45] * len(stressed)))
            )
        )

        stressed['scenario'] = scenario.name
        return stressed

    def run_all_scenarios(self,
                          scenarios: List[StressScenario] = None) -> Dict[str, Dict]:
        """
        Run all predefined stress scenarios and compile results.
        Returns dictionary of scenario results with key metrics.
        """
        if scenarios is None:
            scenarios = PREDEFINED_SCENARIOS

        results = {}
        for scenario in scenarios:
            stressed_df = self.apply_scenario(scenario)
            total_stressed_el = stressed_df['stressed_el'].sum()
            el_increase = total_stressed_el - self.base_expected_loss
            el_increase_pct = (el_increase / max(self.base_expected_loss, 1)) * 100

            # Top impacted companies
            top_impacted = stressed_df.nlargest(5, 'stressed_el')[[
                'company_name', 'sector', 'exposure_usd', 'stressed_el'
            ]].to_dict('records') if 'company_name' in stressed_df.columns else []

            results[scenario.name] = {
                'scenario_description': scenario.description,
                'rate_shock_bps': scenario.rate_shock_bps,
                'gdp_contraction_pct': scenario.gdp_contraction_pct,
                'base_expected_loss': round(self.base_expected_loss, 2),
                'stressed_expected_loss': round(total_stressed_el, 2),
                'el_increase_usd': round(el_increase, 2),
                'el_increase_pct': round(el_increase_pct, 2),
                'stressed_el_as_pct_exposure': round(
                    (total_stressed_el / self.total_exposure) * 100, 4
                ),
                'top_5_impacted': top_impacted,
                'capital_adequacy_impact': self._assess_capital_adequacy(total_stressed_el)
            }

        return results

    def _assess_capital_adequacy(self, stressed_el: float) -> str:
        """Assess capital adequacy under stressed conditions."""
        stressed_el_ratio = stressed_el / self.total_exposure
        if stressed_el_ratio > 0.10:
            return 'CRITICAL - Capital buffer likely breached'
        elif stressed_el_ratio > 0.05:
            return 'WARNING - Capital buffer under pressure'
        elif stressed_el_ratio > 0.02:
            return 'CAUTION - Monitor closely'
        return 'ADEQUATE - Within acceptable range'

    def sensitivity_analysis(self, pd_range: List[float] = None,
                              lgd_range: List[float] = None) -> pd.DataFrame:
        """
        Sensitivity analysis: compute portfolio EL across PD and LGD multiplier grids.
        Returns a matrix of stressed EL values.
        """
        if pd_range is None:
            pd_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        if lgd_range is None:
            lgd_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

        matrix = []
        for pd_mult in pd_range:
            row = {}
            for lgd_adj in lgd_range:
                temp_scenario = StressScenario(
                    name='sensitivity',
                    description='',
                    rate_shock_bps=0,
                    gdp_contraction_pct=0,
                    default_rate_multiplier=pd_mult,
                    lgd_adjustment=lgd_adj
                )
                stressed = self.apply_scenario(temp_scenario)
                row[f'LGD+{int(lgd_adj*100)}bps'] = round(
                    stressed['stressed_el'].sum() / self.total_exposure * 100, 4
                )
            row['PD_Multiplier'] = pd_mult
            matrix.append(row)

        df = pd.DataFrame(matrix).set_index('PD_Multiplier')
        return df

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a concise scenario comparison summary table.
        """
        results = self.run_all_scenarios()
        rows = []
        for scenario_name, data in results.items():
            rows.append({
                'Scenario': scenario_name,
                'Rate Shock (bps)': data['rate_shock_bps'],
                'GDP Contraction (%)': data['gdp_contraction_pct'],
                'Base EL ($M)': round(data['base_expected_loss'] / 1e6, 2),
                'Stressed EL ($M)': round(data['stressed_expected_loss'] / 1e6, 2),
                'EL Increase ($M)': round(data['el_increase_usd'] / 1e6, 2),
                'EL Increase (%)': data['el_increase_pct'],
                'Capital Assessment': data['capital_adequacy_impact']
            })
        return pd.DataFrame(rows)
