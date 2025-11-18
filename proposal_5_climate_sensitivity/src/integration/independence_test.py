"""
Independence Testing for Climate Constraints

Tests whether constraints are truly independent to avoid double-counting.
"""

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, Tuple


class IndependenceTest:
    """
    Test independence between climate sensitivity constraints

    Methods:
    --------
    - Correlation analysis
    - Mutual information
    - Partial correlation
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def test_sample_independence(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray
    ) -> Dict:
        """
        Test independence between two sets of ECS samples

        Parameters:
        -----------
        samples1 : np.ndarray
            First constraint's ECS samples
        samples2 : np.ndarray
            Second constraint's ECS samples

        Returns:
        --------
        results : dict
            Independence test results
        """
        # Correlation test
        correlation, p_value = stats.pearsonr(samples1, samples2)

        # Mutual information
        mi = mutual_info_regression(
            samples1.reshape(-1, 1),
            samples2,
            random_state=42
        )[0]

        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pval = stats.ks_2samp(samples1, samples2)

        # Independence assessment
        independent = (
            abs(correlation) < 0.3 and
            p_value > self.significance_level and
            mi < 0.2
        )

        results = {
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'mutual_information': mi,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'independent': independent,
            'interpretation': self._interpret_independence(
                correlation, mi, independent
            )
        }

        return results

    def _interpret_independence(
        self,
        correlation: float,
        mi: float,
        independent: bool
    ) -> str:
        """
        Provide human-readable interpretation

        Parameters:
        -----------
        correlation : float
            Correlation coefficient
        mi : float
            Mutual information
        independent : bool
            Independence flag

        Returns:
        --------
        interpretation : str
            Text interpretation
        """
        if independent:
            return "Constraints appear independent - safe to combine"
        elif abs(correlation) >= 0.5:
            return f"Strong correlation ({correlation:.2f}) - constraints may not be independent"
        elif mi >= 0.3:
            return f"High mutual information ({mi:.2f}) - shared information detected"
        else:
            return "Weak dependence detected - combine with caution"

    def calculate_partial_correlation(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray,
        control: np.ndarray
    ) -> float:
        """
        Calculate partial correlation controlling for a third variable

        Parameters:
        -----------
        samples1, samples2 : np.ndarray
            Samples to test
        control : np.ndarray
            Variable to control for (e.g., true ECS in models)

        Returns:
        --------
        partial_corr : float
            Partial correlation coefficient
        """
        # Residualize both samples against control
        from sklearn.linear_model import LinearRegression

        lr1 = LinearRegression()
        lr1.fit(control.reshape(-1, 1), samples1)
        residuals1 = samples1 - lr1.predict(control.reshape(-1, 1))

        lr2 = LinearRegression()
        lr2.fit(control.reshape(-1, 1), samples2)
        residuals2 = samples2 - lr2.predict(control.reshape(-1, 1))

        # Correlation of residuals
        partial_corr = np.corrcoef(residuals1, residuals2)[0, 1]

        return partial_corr

    def __repr__(self) -> str:
        return f"IndependenceTest(significance_level={self.significance_level})"
