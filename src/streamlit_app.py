import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import local modules (adjust paths if running from different directory)
try:
    from data_generator import CreditPortfolioGenerator
    from concentration_risk import ConcentrationRiskAnalyzer
    from var_model import VaRModel
    from scenario_analysis import ScenarioAnalyzer, PREDEFINED_SCENARIOS
    from ecl_model import ECLCalculator
except ImportError:
    st.error("Unable to import required modules. Ensure all source files are in the same directory.")


st.set_page_config(
    page_title="Credit Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical { color: #d32f2f; font-weight: bold; }
    .alert-warning { color: #f57c00; font-weight: bold; }
    .alert-ok { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


st.title("📊 Credit Analytics Engine — Institutional Portfolio Risk Dashboard")
st.markdown("**Enterprise-grade credit risk analysis for institutional portfolios**")
st.markdown("---")


# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

portfolio_size = st.sidebar.slider(
    "Portfolio Size (# Companies)",
    min_value=50, max_value=200, value=100, step=10
)

total_exposure_mm = st.sidebar.number_input(
    "Total Exposure ($MM)",
    min_value=100, max_value=10000, value=1000, step=100
)

risk_seed = st.sidebar.number_input(
    "Random Seed (Reproducibility)",
    min_value=1, max_value=9999, value=42
)

if st.sidebar.button("🔄 Generate New Portfolio"):
    st.session_state.clear()
    st.rerun()


# Generate or load portfolio
@st.cache_data
def load_portfolio(n_companies, total_exp, seed):
    gen = CreditPortfolioGenerator(n_companies=n_companies, random_seed=seed)
    portfolio = gen.generate_portfolio()
    # Scale to desired exposure
    scale_factor = (total_exp * 1_000_000) / portfolio['exposure_usd'].sum()
    portfolio['exposure_usd'] = portfolio['exposure_usd'] * scale_factor
    portfolio['weight'] = portfolio['exposure_usd'] / portfolio['exposure_usd'].sum()
    return portfolio


portfolio_df = load_portfolio(portfolio_size, total_exposure_mm, risk_seed)


# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Portfolio Overview",
    "🎯 Concentration Risk",
    "⚠️ VaR Analysis",
    "💥 Stress Testing",
    "📄 Executive Summary"
])


# TAB 1: Portfolio Overview
with tab1:
    st.header("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Exposure", f"${portfolio_df['exposure_usd'].sum()/1e6:.1f}M")
    with col2:
        st.metric("# Companies", len(portfolio_df))
    with col3:
        avg_rating = portfolio_df['rating_score'].mean() if 'rating_score' in portfolio_df.columns else 0
        st.metric("Avg Credit Score", f"{avg_rating:.1f}")
    with col4:
        st.metric("# Sectors", portfolio_df['sector'].nunique())

    st.subheader("Sector Allocation")
    sector_exp = portfolio_df.groupby('sector')['exposure_usd'].sum().sort_values(ascending=False)
    fig_sector = px.pie(
        values=sector_exp.values,
        names=sector_exp.index,
        title="Exposure by Sector",
        hole=0.4
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    st.subheader("Top 10 Exposures")
    top10 = portfolio_df.nlargest(10, 'exposure_usd')[['company_name', 'sector', 'exposure_usd', 'weight']]
    top10['exposure_usd'] = top10['exposure_usd'].apply(lambda x: f"${x/1e6:.2f}M")
    top10['weight'] = top10['weight'].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(top10, use_container_width=True, hide_index=True)


# TAB 2: Concentration Risk
with tab2:
    st.header("Concentration Risk Analysis")

    conc_analyzer = ConcentrationRiskAnalyzer(portfolio_df)
    summary = conc_analyzer.generate_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        hhi_class = summary['concentration_level']
        color = 'alert-critical' if hhi_class == 'HIGH' else ('alert-warning' if hhi_class == 'MODERATE' else 'alert-ok')
        st.markdown(f"**Portfolio HHI**: <span class='{color}'>{summary['portfolio_hhi']:.4f} ({hhi_class})</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Sector HHI", f"{summary['sector_hhi']:.4f}")
    with col3:
        st.metric("Top 10 Concentration", f"{summary['top10_concentration']*100:.1f}%")

    st.subheader("Basel III Single-Name Limit Breaches")
    breaches = conc_analyzer.single_name_breaches()
    if len(breaches) > 0:
        st.warning(f"⚠️ {len(breaches)} positions exceed the 10% single-name limit")
        st.dataframe(breaches, use_container_width=True)
    else:
        st.success("✅ No Basel III single-name limit breaches")

    st.subheader("Sector Concentration Report")
    sector_report = conc_analyzer.sector_concentration_report()
    st.dataframe(sector_report, use_container_width=True)


# TAB 3: VaR Analysis
with tab3:
    st.header("Value-at-Risk (VaR) Analysis")

    st.info("Note: Historical returns are simulated for demo purposes. Use actual market data in production.")

    # Simulate returns
    np.random.seed(risk_seed)
    returns = pd.DataFrame(
        np.random.randn(252, len(portfolio_df)) * 0.02,
        columns=range(len(portfolio_df))
    )

    var_model = VaRModel(portfolio_df, returns)

    confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.99)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Historical VaR")
        hist_var = var_model.historical_var(confidence)
        st.metric("VaR (USD)", f"${hist_var['var_usd']/1e6:.2f}M")
        st.metric("Expected Shortfall", f"${hist_var['es_usd']/1e6:.2f}M")

    with col2:
        st.subheader("Parametric VaR")
        param_var = var_model.parametric_var(confidence)
        st.metric("1-Day VaR", f"${param_var['var_usd_1day']/1e6:.2f}M")
        st.metric("10-Day VaR (Basel)", f"${param_var['var_usd_10day']/1e6:.2f}M")

    with col3:
        st.subheader("Monte Carlo VaR")
        mc_var = var_model.monte_carlo_var(confidence)
        st.metric("VaR (USD)", f"${mc_var['var_usd']/1e6:.2f}M")
        st.metric("Worst Case", f"${mc_var['worst_case_usd']/1e6:.2f}M")

    st.subheader("VaR Comparison Across Methods")
    var_comparison = var_model.comprehensive_var_report()
    st.dataframe(var_comparison, use_container_width=True)


# TAB 4: Stress Testing
with tab4:
    st.header("Scenario Analysis & Stress Testing")

    scenario_analyzer = ScenarioAnalyzer(portfolio_df)

    st.subheader("Predefined Basel III / EBA Scenarios")
    scenario_results = scenario_analyzer.run_all_scenarios()

    scenario_names = list(scenario_results.keys())
    selected_scenario = st.selectbox("Select Scenario", scenario_names)

    scenario_data = scenario_results[selected_scenario]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{selected_scenario}**")
        st.write(scenario_data['scenario_description'])
        st.metric("Rate Shock (bps)", scenario_data['rate_shock_bps'])
        st.metric("GDP Contraction (%)", scenario_data['gdp_contraction_pct'])

    with col2:
        st.metric("Base Expected Loss", f"${scenario_data['base_expected_loss']/1e6:.2f}M")
        st.metric("Stressed Expected Loss", f"${scenario_data['stressed_expected_loss']/1e6:.2f}M")
        el_delta = scenario_data['el_increase_pct']
        st.metric("EL Increase", f"{el_delta:.1f}%", delta=f"{el_delta:.1f}%")
        status = scenario_data['capital_adequacy_impact']
        if 'CRITICAL' in status:
            st.error(status)
        elif 'WARNING' in status:
            st.warning(status)
        else:
            st.success(status)

    st.subheader("Scenario Comparison")
    comparison_table = scenario_analyzer.generate_summary_table()
    st.dataframe(comparison_table, use_container_width=True)


# TAB 5: Executive Summary
with tab5:
    st.header("Executive Summary Report")

    st.subheader("📌 Key Findings")

    # Portfolio metrics
    st.write(f"**Total Portfolio Exposure**: ${portfolio_df['exposure_usd'].sum()/1e6:.1f}M across {len(portfolio_df)} obligors")

    # Concentration risk
    conc_summary = conc_analyzer.generate_summary()
    st.write(f"**Concentration Risk**: Portfolio HHI = {conc_summary['portfolio_hhi']:.4f} ({conc_summary['concentration_level']})")
    if conc_summary['single_name_breaches'] > 0:
        st.write(f"⚠️ **{conc_summary['single_name_breaches']} Basel III breaches** with ${conc_summary['total_excess_exposure']/1e6:.2f}M excess exposure")

    # VaR
    st.write(f"**99% 1-Day VaR (Parametric)**: ${param_var['var_usd_1day']/1e6:.2f}M ({param_var['var_usd_1day']/portfolio_df['exposure_usd'].sum()*100:.2f}% of portfolio)")

    # Stress test
    worst_scenario = max(scenario_results.items(), key=lambda x: x[1]['el_increase_usd'])
    st.write(f"**Worst Stress Scenario**: {worst_scenario[0]} with EL increase of ${worst_scenario[1]['el_increase_usd']/1e6:.2f}M")

    st.subheader("📊 Risk Heatmap")

    heatmap_data = []
    for name, data in scenario_results.items():
        heatmap_data.append({
            'Scenario': name,
            'Severity': data['el_increase_pct']
        })

    fig_heatmap = px.bar(
        pd.DataFrame(heatmap_data).sort_values('Severity', ascending=False),
        x='Severity',
        y='Scenario',
        orientation='h',
        title="Stress Test Severity (EL % Increase)",
        color='Severity',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    st.caption(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Credit Analytics Engine v1.0")
