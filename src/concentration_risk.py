import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ConcentrationRiskAnalyzer:
    """
    Concentration Risk Analyzer implementing Herfindahl-Hirschman Index (HHI)
    and Basel III single-name exposure limits for institutional credit portfolios.
    """

    BASEL_SINGLE_NAME_LIMIT = 0.10  # 10% single-name exposure limit
    HIGH_CONCENTRATION_HHI = 0.25   # HHI > 0.25 indicates high concentration
    MODERATE_CONCENTRATION_HHI = 0.15  # HHI 0.15-0.25 moderate concentration

    def __init__(self, portfolio_df: pd.DataFrame):
        """
        Initialize with portfolio DataFrame.
        Expected columns: company_id, company_name, sector, exposure_usd, rating
        """
        self.portfolio = portfolio_df.copy()
        self.total_exposure = portfolio_df['exposure_usd'].sum()
        self.portfolio['weight'] = self.portfolio['exposure_usd'] / self.total_exposure

    def compute_hhi(self, group_by: str = 'sector') -> Dict[str, float]:
        """
        Compute Herfindahl-Hirschman Index for portfolio concentration.
        HHI = sum(s_i^2) where s_i is the market share / weight of each entity.
        Returns HHI at portfolio level and by grouping dimension.
        """
        # Portfolio-level HHI (single-name)
        portfolio_hhi = (self.portfolio['weight'] ** 2).sum()

        # Sector/group-level HHI
        group_weights = self.portfolio.groupby(group_by)['exposure_usd'].sum() / self.total_exposure
        group_hhi = (group_weights ** 2).sum()

        return {
            'portfolio_hhi': round(portfolio_hhi, 6),
            f'{group_by}_hhi': round(group_hhi, 6),
            'concentration_level': self._classify_concentration(portfolio_hhi)
        }

    def _classify_concentration(self, hhi: float) -> str:
        if hhi > self.HIGH_CONCENTRATION_HHI:
            return 'HIGH'
        elif hhi > self.MODERATE_CONCENTRATION_HHI:
            return 'MODERATE'
        return 'LOW'

    def single_name_breaches(self) -> pd.DataFrame:
        """
        Identify exposures that breach Basel III single-name concentration limits.
        Returns DataFrame of breaching positions with excess amounts.
        """
        breaches = self.portfolio[self.portfolio['weight'] > self.BASEL_SINGLE_NAME_LIMIT].copy()
        breaches['excess_weight'] = breaches['weight'] - self.BASEL_SINGLE_NAME_LIMIT
        breaches['excess_usd'] = breaches['excess_weight'] * self.total_exposure
        return breaches[['company_id', 'company_name', 'sector', 'exposure_usd',
                          'weight', 'excess_weight', 'excess_usd']].reset_index(drop=True)

    def sector_concentration_report(self) -> pd.DataFrame:
        """
        Generate sector-level concentration report with HHI contribution.
        """
        sector_data = self.portfolio.groupby('sector').agg(
            total_exposure=('exposure_usd', 'sum'),
            num_companies=('company_id', 'count'),
            avg_rating=('rating_score', 'mean') if 'rating_score' in self.portfolio.columns else ('exposure_usd', 'count')
        ).reset_index()

        sector_data['weight'] = sector_data['total_exposure'] / self.total_exposure
        sector_data['hhi_contribution'] = sector_data['weight'] ** 2
        sector_data = sector_data.sort_values('weight', ascending=False)
        sector_data['cumulative_weight'] = sector_data['weight'].cumsum()
        return sector_data

    def top_n_concentration(self, n: int = 10) -> Tuple[pd.DataFrame, float]:
        """
        Compute top-N name concentration and flag if above threshold.
        Returns sorted exposures and the cumulative weight of top N.
        """
        top_n = self.portfolio.nlargest(n, 'exposure_usd')[[
            'company_id', 'company_name', 'sector', 'exposure_usd', 'weight'
        ]].reset_index(drop=True)
        cumulative_weight = top_n['weight'].sum()
        return top_n, round(cumulative_weight, 4)

    def correlation_adjusted_hhi(self, correlation_matrix: pd.DataFrame) -> float:
        """
        Compute correlation-adjusted HHI using sector correlation matrix.
        Provides a more conservative concentration measure.
        HHI_adj = w' * C * w where C is the correlation matrix.
        """
        sector_weights = self.portfolio.groupby('sector')['weight'].sum()
        # Align weights with correlation matrix index
        aligned_weights = sector_weights.reindex(correlation_matrix.index, fill_value=0)
        w = aligned_weights.values
        C = correlation_matrix.values
        hhi_adj = float(w @ C @ w)
        return round(hhi_adj, 6)

    def generate_summary(self) -> Dict:
        """
        Generate a comprehensive concentration risk summary.
        """
        hhi_results = self.compute_hhi()
        breaches = self.single_name_breaches()
        top10, top10_weight = self.top_n_concentration(10)

        return {
            'total_portfolio_exposure': self.total_exposure,
            'num_companies': len(self.portfolio),
            'portfolio_hhi': hhi_results['portfolio_hhi'],
            'sector_hhi': hhi_results['sector_hhi'],
            'concentration_level': hhi_results['concentration_level'],
            'single_name_breaches': len(breaches),
            'total_excess_exposure': breaches['excess_usd'].sum() if len(breaches) > 0 else 0,
            'top10_concentration': top10_weight,
            'rebalancing_required': len(breaches) > 0 or hhi_results['portfolio_hhi'] > self.HIGH_CONCENTRATION_HHI
        }
