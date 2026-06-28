import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO


class CreditRiskReportGenerator:
    """
    Automated PDF report generator for institutional credit risk analytics.
    Produces executive-ready reports with key metrics, charts, and recommendations.
    """

    def __init__(self, portfolio_df: pd.DataFrame,
                 concentration_summary: Dict,
                 var_results: Dict,
                 scenario_results: Dict):
        """
        Initialize report generator with analysis results.
        """
        self.portfolio = portfolio_df
        self.concentration = concentration_summary
        self.var = var_results
        self.scenarios = scenario_results
        self.report_date = datetime.now()

    def generate_executive_summary(self) -> str:
        """
        Generate text-based executive summary with key findings.
        """
        total_exposure = self.portfolio['exposure_usd'].sum()
        num_companies = len(self.portfolio)

        summary = f"""
========================================
CREDIT ANALYTICS ENGINE - EXECUTIVE SUMMARY
========================================

Report Date: {self.report_date.strftime('%B %d, %Y')}
Portfolio Analysis Period: {self.report_date.strftime('%Y-%m-%d')}

----------------------------------------
1. PORTFOLIO OVERVIEW
----------------------------------------
Total Exposure: ${total_exposure/1e6:,.2f} million
Number of Obligors: {num_companies}
Number of Sectors: {self.portfolio['sector'].nunique()}

Top 3 Sector Exposures:
"""

        sector_exp = self.portfolio.groupby('sector')['exposure_usd'].sum().sort_values(ascending=False).head(3)
        for idx, (sector, exp) in enumerate(sector_exp.items(), 1):
            pct = (exp / total_exposure) * 100
            summary += f"  {idx}. {sector}: ${exp/1e6:,.2f}M ({pct:.1f}%)\n"

        # Concentration Risk
        summary += f"""

----------------------------------------
2. CONCENTRATION RISK ANALYSIS
----------------------------------------
Portfolio HHI: {self.concentration['portfolio_hhi']:.4f}
Concentration Level: {self.concentration['concentration_level']}
Top 10 Name Concentration: {self.concentration['top10_concentration']*100:.1f}%

Basel III Compliance:
"""

        if self.concentration['single_name_breaches'] > 0:
            summary += f"  ⚠ WARNING: {self.concentration['single_name_breaches']} single-name limit breaches detected\n"
            summary += f"  Excess Exposure: ${self.concentration['total_excess_exposure']/1e6:,.2f}M\n"
            summary += "  Recommendation: Immediate portfolio rebalancing required\n"
        else:
            summary += "  ✓ All positions within 10% single-name limit\n"

        # VaR Analysis
        summary += f"""

----------------------------------------
3. VALUE-AT-RISK (VaR) METRICS
----------------------------------------
99% Confidence Level:
  - Historical VaR: ${self.var.get('historical_var_usd', 0)/1e6:,.2f}M
  - Parametric VaR (1-day): ${self.var.get('parametric_var_1day_usd', 0)/1e6:,.2f}M
  - Parametric VaR (10-day): ${self.var.get('parametric_var_10day_usd', 0)/1e6:,.2f}M
  - Monte Carlo VaR: ${self.var.get('monte_carlo_var_usd', 0)/1e6:,.2f}M

"""

        # Stress Testing
        worst_scenario = max(self.scenarios.items(), key=lambda x: x[1]['el_increase_usd'])
        summary += f"""
----------------------------------------
4. STRESS TESTING RESULTS
----------------------------------------
Most Severe Scenario: {worst_scenario[0]}
  Rate Shock: {worst_scenario[1]['rate_shock_bps']} bps
  GDP Contraction: {worst_scenario[1]['gdp_contraction_pct']}%
  Expected Loss Increase: ${worst_scenario[1]['el_increase_usd']/1e6:,.2f}M ({worst_scenario[1]['el_increase_pct']:.1f}%)
  Capital Adequacy: {worst_scenario[1]['capital_adequacy_impact']}

"""

        # Recommendations
        summary += """
----------------------------------------
5. KEY RECOMMENDATIONS
----------------------------------------
"""

        recommendations = []

        if self.concentration['concentration_level'] == 'HIGH':
            recommendations.append("  • URGENT: Reduce portfolio concentration through strategic diversification")

        if self.concentration['single_name_breaches'] > 0:
            recommendations.append("  • CRITICAL: Rebalance positions exceeding Basel III single-name limits")

        if self.concentration['top10_concentration'] > 0.50:
            recommendations.append("  • WARNING: Top 10 names exceed 50% of portfolio - consider tail risk")

        # Check stress test severity
        critical_scenarios = [name for name, data in self.scenarios.items()
                               if 'CRITICAL' in data['capital_adequacy_impact']]
        if critical_scenarios:
            recommendations.append(f"  • STRESS TEST ALERT: {len(critical_scenarios)} scenario(s) breach capital adequacy")

        if not recommendations:
            recommendations.append("  • Portfolio risk metrics within acceptable ranges")
            recommendations.append("  • Continue monitoring concentration and stress scenarios quarterly")

        summary += "\n".join(recommendations)

        summary += f"""

========================================
End of Executive Summary
Generated by Credit Analytics Engine v1.0
========================================
"""

        return summary

    def generate_sector_chart(self) -> BytesIO:
        """
        Generate sector allocation pie chart.
        Returns BytesIO object for PDF embedding or display.
        """
        sector_exp = self.portfolio.groupby('sector')['exposure_usd'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3.colors
        wedges, texts, autotexts = ax.pie(
            sector_exp.values,
            labels=sector_exp.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('Portfolio Exposure by Sector', fontsize=14, fontweight='bold')

        # Style
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    def generate_stress_test_chart(self) -> BytesIO:
        """
        Generate stress test comparison bar chart.
        """
        scenario_data = []
        for name, data in self.scenarios.items():
            scenario_data.append({
                'Scenario': name,
                'EL_Increase_Pct': data['el_increase_pct']
            })

        df = pd.DataFrame(scenario_data).sort_values('EL_Increase_Pct', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d32f2f' if x > 200 else '#f57c00' if x > 100 else '#388e3c'
                  for x in df['EL_Increase_Pct']]

        ax.barh(df['Scenario'], df['EL_Increase_Pct'], color=colors)
        ax.set_xlabel('Expected Loss Increase (%)', fontsize=12, fontweight='bold')
        ax.set_title('Stress Test Scenario Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Legend
        red_patch = mpatches.Patch(color='#d32f2f', label='Severe (>200%)')
        orange_patch = mpatches.Patch(color='#f57c00', label='Moderate (100-200%)')
        green_patch = mpatches.Patch(color='#388e3c', label='Mild (<100%)')
        ax.legend(handles=[red_patch, orange_patch, green_patch], loc='lower right')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    def save_to_text_file(self, filename: str = 'credit_risk_report.txt'):
        """
        Save executive summary to text file.
        """
        summary = self.generate_executive_summary()
        with open(filename, 'w') as f:
            f.write(summary)
        return filename

    def export_data_tables(self, output_dir: str = './reports') -> Dict[str, str]:
        """
        Export key data tables to CSV files for detailed analysis.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        files_created = {}

        # Portfolio export
        portfolio_file = f"{output_dir}/portfolio_snapshot_{self.report_date.strftime('%Y%m%d')}.csv"
        self.portfolio.to_csv(portfolio_file, index=False)
        files_created['portfolio'] = portfolio_file

        # Top exposures
        top20_file = f"{output_dir}/top20_exposures_{self.report_date.strftime('%Y%m%d')}.csv"
        self.portfolio.nlargest(20, 'exposure_usd').to_csv(top20_file, index=False)
        files_created['top_exposures'] = top20_file

        # Sector breakdown
        sector_file = f"{output_dir}/sector_breakdown_{self.report_date.strftime('%Y%m%d')}.csv"
        sector_summary = self.portfolio.groupby('sector').agg(
            total_exposure=('exposure_usd', 'sum'),
            num_companies=('company_id', 'count')
        ).reset_index()
        sector_summary.to_csv(sector_file, index=False)
        files_created['sector_breakdown'] = sector_file

        return files_created
