import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class VaRModel:
    """
    Value-at-Risk (VaR) Model implementing Historical Simulation,
    Parametric (Variance-Covariance), and Monte Carlo methods
    for institutional credit portfolio risk measurement.
    """

    CONFIDENCE_LEVELS = [0.95, 0.99]
    MC_ITERATIONS = 10_000
    HOLDING_PERIOD_DAYS = 10  # Basel III standard

    def __init__(self, portfolio_df: pd.DataFrame, returns_df: pd.DataFrame = None):
        """
        Initialize VaR model.
        portfolio_df: Portfolio with exposure_usd and weight columns.
        returns_df: Historical returns DataFrame (companies as columns).
        """
        self.portfolio = portfolio_df.copy()
        self.returns = returns_df
        self.total_exposure = portfolio_df['exposure_usd'].sum()
        self.weights = portfolio_df['weight'].values if 'weight' in portfolio_df.columns else (
            portfolio_df['exposure_usd'].values / self.total_exposure
        )

    def historical_var(self, confidence: float = 0.99) -> Dict[str, float]:
        """
        Historical Simulation VaR.
        Uses actual historical P&L distribution without distributional assumptions.
        """
        if self.returns is None:
            raise ValueError("Historical returns data required for historical VaR.")

        portfolio_returns = self.returns.dot(self.weights)
        var_level = np.percentile(portfolio_returns, (1 - confidence) * 100)
        es_level = portfolio_returns[portfolio_returns <= var_level].mean()

        return {
            'method': 'Historical Simulation',
            'confidence': confidence,
            'var_pct': round(var_level, 6),
            'var_usd': round(abs(var_level) * self.total_exposure, 2),
            'es_pct': round(es_level, 6),
            'es_usd': round(abs(es_level) * self.total_exposure, 2)
        }

    def parametric_var(self, confidence: float = 0.99,
                       cov_matrix: np.ndarray = None) -> Dict[str, float]:
        """
        Parametric (Variance-Covariance) VaR.
        Assumes normally distributed returns.
        """
        if cov_matrix is None and self.returns is not None:
            cov_matrix = self.returns.cov().values
        elif cov_matrix is None:
            raise ValueError("Covariance matrix or returns data required.")

        portfolio_variance = self.weights @ cov_matrix @ self.weights
        portfolio_std = np.sqrt(portfolio_variance)
        z_score = stats.norm.ppf(1 - confidence)

        var_pct = z_score * portfolio_std
        es_pct = -portfolio_std * stats.norm.pdf(stats.norm.ppf(1 - confidence)) / (1 - confidence)

        # Scale to holding period
        var_hp = var_pct * np.sqrt(self.HOLDING_PERIOD_DAYS)
        es_hp = es_pct * np.sqrt(self.HOLDING_PERIOD_DAYS)

        return {
            'method': 'Parametric (Variance-Covariance)',
            'confidence': confidence,
            'portfolio_volatility': round(portfolio_std, 6),
            'var_pct_1day': round(abs(var_pct), 6),
            'var_usd_1day': round(abs(var_pct) * self.total_exposure, 2),
            'var_pct_10day': round(abs(var_hp), 6),
            'var_usd_10day': round(abs(var_hp) * self.total_exposure, 2),
            'es_usd_1day': round(abs(es_pct) * self.total_exposure, 2)
        }

    def monte_carlo_var(self, confidence: float = 0.99,
                        cov_matrix: np.ndarray = None,
                        mean_returns: np.ndarray = None) -> Dict[str, float]:
        """
        Monte Carlo Simulation VaR with 10,000 scenarios.
        Simulates correlated asset returns using Cholesky decomposition.
        """
        if cov_matrix is None and self.returns is not None:
            cov_matrix = self.returns.cov().values
        elif cov_matrix is None:
            raise ValueError("Covariance matrix required for Monte Carlo VaR.")

        if mean_returns is None:
            if self.returns is not None:
                mean_returns = self.returns.mean().values
            else:
                mean_returns = np.zeros(len(self.weights))

        n_assets = len(self.weights)
        np.random.seed(42)

        # Cholesky decomposition for correlated sampling
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization if matrix is not positive definite
            cov_matrix += np.eye(n_assets) * 1e-8
            L = np.linalg.cholesky(cov_matrix)

        Z = np.random.standard_normal((self.MC_ITERATIONS, n_assets))
        simulated_returns = mean_returns + Z @ L.T
        portfolio_sim_returns = simulated_returns @ self.weights

        var_pct = np.percentile(portfolio_sim_returns, (1 - confidence) * 100)
        es_pct = portfolio_sim_returns[portfolio_sim_returns <= var_pct].mean()

        return {
            'method': 'Monte Carlo Simulation',
            'confidence': confidence,
            'iterations': self.MC_ITERATIONS,
            'var_pct': round(var_pct, 6),
            'var_usd': round(abs(var_pct) * self.total_exposure, 2),
            'es_pct': round(es_pct, 6),
            'es_usd': round(abs(es_pct) * self.total_exposure, 2),
            'worst_case_usd': round(abs(portfolio_sim_returns.min()) * self.total_exposure, 2)
        }

    def comprehensive_var_report(self, cov_matrix: np.ndarray = None) -> pd.DataFrame:
        """
        Generate comprehensive VaR report across all methods and confidence levels.
        """
        results = []
        for conf in self.CONFIDENCE_LEVELS:
            try:
                hist = self.historical_var(conf)
                results.append({'Method': hist['method'], 'Confidence': conf,
                                 'VaR USD': hist['var_usd'], 'ES USD': hist['es_usd']})
            except Exception:
                pass

            try:
                param = self.parametric_var(conf, cov_matrix)
                results.append({'Method': param['method'], 'Confidence': conf,
                                 'VaR USD': param['var_usd_1day'], 'ES USD': param['es_usd_1day']})
            except Exception:
                pass

            try:
                mc = self.monte_carlo_var(conf, cov_matrix)
                results.append({'Method': mc['method'], 'Confidence': conf,
                                 'VaR USD': mc['var_usd'], 'ES USD': mc['es_usd']})
            except Exception:
                pass

        return pd.DataFrame(results)

    def backtesting_pof_test(self, actual_returns: np.ndarray,
                             var_estimates: np.ndarray,
                             confidence: float = 0.99) -> Dict:
        """
        Kupiec Proportion of Failures (POF) test for VaR backtesting.
        Tests if the number of VaR breaches is statistically acceptable.
        """
        n = len(actual_returns)
        exceptions = np.sum(actual_returns < -var_estimates)
        exception_rate = exceptions / n
        expected_rate = 1 - confidence

        # Kupiec likelihood ratio statistic
        if exceptions == 0:
            lr_stat = -2 * n * np.log(1 - expected_rate)
        elif exceptions == n:
            lr_stat = -2 * n * np.log(expected_rate)
        else:
            lr_stat = -2 * (np.log((1 - expected_rate) ** (n - exceptions) * expected_rate ** exceptions) -
                            np.log((1 - exception_rate) ** (n - exceptions) * exception_rate ** exceptions))

        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        passed = p_value > 0.05

        return {
            'n_observations': n,
            'n_exceptions': int(exceptions),
            'exception_rate': round(exception_rate, 4),
            'expected_rate': expected_rate,
            'lr_statistic': round(lr_stat, 4),
            'p_value': round(p_value, 4),
            'test_passed': passed,
            'verdict': 'PASS - Model is accurate' if passed else 'FAIL - Review VaR model'
        }
