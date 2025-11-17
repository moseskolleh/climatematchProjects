"""
Unit tests for causal discovery methods
"""

import pytest
import numpy as np
from src.discovery.causal_discovery import (
    CausalDiscoveryEngine,
    GrangerCausality,
    CausalityResult
)


class TestGrangerCausality:
    """Tests for Granger causality method"""

    def test_granger_with_known_relationship(self):
        """Test Granger causality detects known lagged relationship"""
        # Create synthetic data with known causal relationship
        np.random.seed(42)
        n = 200

        # X causes Y with 3-month lag
        X = np.random.randn(n)
        Y = np.zeros(n)
        Y[3:] = 0.7 * X[:-3] + 0.3 * np.random.randn(n-3)

        # Test
        gc = GrangerCausality({'max_lag': 12})
        result = gc.test_causality(X, Y)

        # Assertions
        assert isinstance(result, CausalityResult)
        assert result.method == 'granger'
        assert result.significant == True
        assert result.optimal_lag == 3
        assert result.p_value < 0.01

    def test_granger_with_no_relationship(self):
        """Test Granger causality rejects when no relationship"""
        # Create independent time series
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        Y = np.random.randn(n)

        # Test
        gc = GrangerCausality({'max_lag': 12})
        result = gc.test_causality(X, Y)

        # Should not find significant relationship
        assert result.significant == False
        assert result.p_value > 0.01


class TestCausalDiscoveryEngine:
    """Tests for the overall causal discovery engine"""

    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = CausalDiscoveryEngine()

        assert 'granger' in engine.methods
        assert 'ccm' in engine.methods
        assert 'transfer_entropy' in engine.methods

    def test_consensus_detection(self):
        """Test consensus is detected when multiple methods agree"""
        # Create data with clear causal relationship
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        Y = np.zeros(n)
        Y[2:] = 0.8 * X[:-2] + 0.2 * np.random.randn(n-2)

        # Run discovery
        engine = CausalDiscoveryEngine()
        results = engine.discover(X, Y)

        # Check structure
        assert hasattr(results, 'consensus')
        assert hasattr(results, 'individual_results')
        assert hasattr(results, 'confidence')

        # At least Granger should detect it
        assert 'granger' in results.individual_results


def test_synthetic_enso_sahel():
    """
    Test with synthetic ENSO-Sahel-like relationship
    This mimics the known anti-correlation between ENSO and Sahel rainfall
    """
    np.random.seed(42)
    n = 500  # ~40 years of monthly data

    # Synthetic ENSO index (Nino3.4-like)
    enso = np.random.randn(n)
    enso = np.convolve(enso, np.ones(5)/5, mode='same')  # Smooth

    # Synthetic Sahel rainfall with 2-month lag and negative correlation
    sahel = np.zeros(n)
    sahel[2:] = -0.6 * enso[:-2] + 0.4 * np.random.randn(n-2)

    # Discover relationship
    engine = CausalDiscoveryEngine()
    results = engine.discover(enso, sahel)

    # Should find significant relationship
    gc_result = results.individual_results['granger']
    assert gc_result.significant == True
    assert gc_result.optimal_lag in [1, 2, 3]  # Should be around 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
