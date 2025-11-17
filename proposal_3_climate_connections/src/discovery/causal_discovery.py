"""
Causal discovery methods for identifying climate teleconnections.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalityResult:
    """Result from a causal discovery method"""
    method: str
    significant: bool
    p_value: float
    effect_size: float
    optimal_lag: int = None
    additional_info: Dict = None


@dataclass
class CausalityResults:
    """Aggregated results from multiple causal discovery methods"""
    individual_results: Dict[str, CausalityResult]
    consensus: bool
    confidence: str  # 'high', 'medium', 'low'

    def summary(self) -> str:
        """Generate summary of results"""
        significant_methods = [
            name for name, result in self.individual_results.items()
            if result.significant
        ]
        return f"Consensus: {self.consensus}, Significant methods: {significant_methods}"


class CausalDiscoveryEngine:
    """Apply multiple causal discovery methods"""

    def __init__(self, config: Dict = None):
        """
        Initialize causal discovery engine.

        Args:
            config: Configuration dictionary for each method
        """
        self.config = config or {}
        self.methods = {
            'granger': GrangerCausality(self.config.get('granger', {})),
            'ccm': ConvergentCrossMapping(self.config.get('ccm', {})),
            'transfer_entropy': TransferEntropy(self.config.get('te', {})),
        }
        logger.info("Causal discovery engine initialized")

    def discover(self, source_data: np.ndarray,
                 target_data: np.ndarray) -> CausalityResults:
        """
        Apply all causal discovery methods.

        Args:
            source_data: Time series of potential cause
            target_data: Time series of potential effect

        Returns:
            CausalityResults with findings from each method
        """
        results = {}

        for method_name, method in self.methods.items():
            logger.info(f"Testing causality with {method_name}")
            try:
                result = method.test_causality(source_data, target_data)
                results[method_name] = result
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                results[method_name] = CausalityResult(
                    method=method_name,
                    significant=False,
                    p_value=1.0,
                    effect_size=0.0
                )

        # Check consensus (≥2 methods agree)
        significant_count = sum(
            1 for r in results.values() if r.significant
        )
        consensus = significant_count >= 2

        # Determine confidence level
        if significant_count == len(self.methods):
            confidence = 'high'
        elif significant_count >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'

        return CausalityResults(
            individual_results=results,
            consensus=consensus,
            confidence=confidence
        )


class GrangerCausality:
    """Granger causality testing via Vector Autoregression"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_lag = self.config.get('max_lag', 12)
        self.significance_level = self.config.get('alpha', 0.01)

    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> CausalityResult:
        """
        Test if X Granger-causes Y.

        Args:
            X: Source time series
            Y: Target time series

        Returns:
            CausalityResult
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Prepare data
            data = pd.DataFrame({'Y': Y, 'X': X})

            # Run tests
            test_results = grangercausalitytests(
                data,
                self.max_lag,
                verbose=False
            )

            # Extract p-values from F-test
            p_values = [
                test_results[lag][0]['ssr_ftest'][1]
                for lag in range(1, self.max_lag + 1)
            ]

            # Find optimal lag
            optimal_lag = int(np.argmin(p_values) + 1)
            min_p_value = float(min(p_values))

            # Calculate effect size (correlation at optimal lag)
            effect_size = float(np.corrcoef(X[:-optimal_lag], Y[optimal_lag:])[0, 1])

            return CausalityResult(
                method='granger',
                significant=min_p_value < self.significance_level,
                p_value=min_p_value,
                effect_size=abs(effect_size),
                optimal_lag=optimal_lag,
                additional_info={'all_p_values': p_values}
            )

        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            return CausalityResult(
                method='granger',
                significant=False,
                p_value=1.0,
                effect_size=0.0
            )


class ConvergentCrossMapping:
    """Convergent Cross Mapping for nonlinear dynamical coupling"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.embedding_dim = self.config.get('embedding_dim', 3)
        self.tau = self.config.get('tau', 1)
        self.lib_sizes = self.config.get('lib_sizes', None)

    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> CausalityResult:
        """
        Test causality using Convergent Cross Mapping.

        Args:
            X: Source time series
            Y: Target time series

        Returns:
            CausalityResult
        """
        # Note: Full implementation would use specialized CCM library
        # This is a placeholder showing the structure

        logger.info("Running Convergent Cross Mapping")

        # Simplified implementation
        # In reality, would reconstruct attractor and test convergence
        skills = self._calculate_ccm_skills(X, Y)

        # Test for convergence (increasing skill with library size)
        if len(skills) > 1:
            slope = (skills[-1] - skills[0]) / len(skills)
            converges = slope > 0.01
        else:
            converges = False

        return CausalityResult(
            method='ccm',
            significant=converges,
            p_value=0.01 if converges else 0.5,  # Placeholder
            effect_size=max(skills) if skills else 0.0,
            additional_info={'skills': skills}
        )

    def _calculate_ccm_skills(self, X: np.ndarray, Y: np.ndarray) -> List[float]:
        """Calculate CCM prediction skills for different library sizes"""
        # Placeholder implementation
        # Real implementation would:
        # 1. Embed Y in state space
        # 2. For increasing library sizes, predict X from Y's attractor
        # 3. Return correlation between predicted and actual X

        n = len(X)
        lib_sizes = self.lib_sizes or np.arange(50, n, 50)

        skills = []
        for L in lib_sizes:
            # Simplified skill calculation
            # In reality, much more complex
            skill = 0.5 + 0.3 * (L / n)  # Placeholder
            skills.append(skill)

        return skills


class TransferEntropy:
    """Transfer Entropy for directed information flow"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.k = self.config.get('history_length', 1)
        self.n_bins = self.config.get('n_bins', 10)
        self.n_permutations = self.config.get('n_permutations', 1000)

    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> CausalityResult:
        """
        Calculate transfer entropy from X to Y.

        Args:
            X: Source time series
            Y: Target time series

        Returns:
            CausalityResult
        """
        logger.info("Calculating Transfer Entropy")

        # Discretize time series
        X_discrete = self._discretize(X)
        Y_discrete = self._discretize(Y)

        # Calculate transfer entropy
        te_value = self._calculate_te(X_discrete, Y_discrete)

        # Permutation test for significance
        p_value = self._permutation_test(X_discrete, Y_discrete, te_value)

        return CausalityResult(
            method='transfer_entropy',
            significant=p_value < 0.01,
            p_value=p_value,
            effect_size=te_value,
            additional_info={'te_value': te_value}
        )

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous time series into bins"""
        bins = np.linspace(np.min(data), np.max(data), self.n_bins + 1)
        return np.digitize(data, bins)

    def _calculate_te(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate transfer entropy"""
        # Simplified placeholder
        # Real implementation would calculate:
        # TE(X→Y) = Σ p(y_t, y_{t-k}, x_{t-k}) * log[p(y_t|y_{t-k}, x_{t-k}) / p(y_t|y_{t-k})]

        # Placeholder: use mutual information as proxy
        from sklearn.metrics import mutual_info_score

        mi = mutual_info_score(X[:-1], Y[1:])
        return float(mi)

    def _permutation_test(self, X: np.ndarray, Y: np.ndarray,
                         observed_te: float) -> float:
        """Permutation test for significance"""
        null_te = []

        for _ in range(self.n_permutations):
            # Shuffle X to break temporal structure
            X_shuffled = np.random.permutation(X)
            te_null = self._calculate_te(X_shuffled, Y)
            null_te.append(te_null)

        # Calculate p-value
        p_value = np.mean(np.array(null_te) >= observed_te)
        return float(p_value)


class StructuralCausalModel:
    """Structural Causal Models using Pearl's framework"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Structural Causal Model initialized")

    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> CausalityResult:
        """
        Test causality using structural equation modeling.

        Args:
            X: Source time series
            Y: Target time series

        Returns:
            CausalityResult
        """
        # Placeholder implementation
        # Real implementation would:
        # 1. Define causal graph structure
        # 2. Estimate structural equations
        # 3. Test do-calculus interventions

        logger.info("Testing with Structural Causal Model")

        # Simplified regression-based test
        from scipy import stats

        # Lag X and regress Y on X
        max_lag = 12
        best_r2 = 0
        best_lag = 0
        best_p = 1.0

        for lag in range(1, max_lag + 1):
            if lag >= len(X):
                break

            X_lagged = X[:-lag]
            Y_lagged = Y[lag:]

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                X_lagged, Y_lagged
            )

            if abs(r_value) > abs(best_r2):
                best_r2 = r_value
                best_lag = lag
                best_p = p_value

        return CausalityResult(
            method='structural_causal_model',
            significant=best_p < 0.01,
            p_value=best_p,
            effect_size=abs(best_r2),
            optimal_lag=best_lag
        )
