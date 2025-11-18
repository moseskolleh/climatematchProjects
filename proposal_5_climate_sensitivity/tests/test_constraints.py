"""
Unit tests for constraint implementations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from constraints.paleoclimate import LGMConstraint, PlioceneConstraint
from constraints.observational import EnergyBudgetConstraint, EmergentConstraint
from constraints.process_based import CloudFeedbackConstraint


class TestLGMConstraint:
    """Test LGM paleoclimate constraint."""

    def test_initialization(self):
        """Test LGM constraint initialization."""
        lgm = LGMConstraint(apply_state_dependence=True, n_samples=1000)
        assert lgm.reference_period == "Last Glacial Maximum"
        assert lgm.n_samples == 1000
        assert lgm.apply_state_dependence is True

    def test_ecs_estimation(self):
        """Test ECS estimation from LGM data."""
        lgm = LGMConstraint(n_samples=1000)

        result = lgm.estimate_ecs(
            temperature_change=-6.0,
            temperature_uncertainty=0.5,
            forcing_change=-7.5,
            forcing_uncertainty=1.0,
            correlation=-0.2
        )

        # Check that result is physically reasonable
        assert result.ecs_mean > 0
        assert result.ecs_mean < 10
        assert result.ecs_std > 0
        assert len(result.ecs_samples) > 0
        assert result.state_dependence_factor > 0

    def test_state_dependence_correction(self):
        """Test that state-dependence correction is applied."""
        lgm_with_correction = LGMConstraint(apply_state_dependence=True, n_samples=1000)
        lgm_without_correction = LGMConstraint(apply_state_dependence=False, n_samples=1000)

        # Use same inputs
        kwargs = {
            "temperature_change": -6.0,
            "temperature_uncertainty": 0.3,
            "forcing_change": -7.5,
            "forcing_uncertainty": 0.8,
            "correlation": 0.0
        }

        result_with = lgm_with_correction.estimate_ecs(**kwargs)
        result_without = lgm_without_correction.estimate_ecs(**kwargs)

        # With correction should give different (higher) ECS
        assert result_with.ecs_mean != result_without.ecs_mean
        assert result_with.state_dependence_factor != 1.0
        assert result_without.state_dependence_factor == 1.0


class TestEnergyBudgetConstraint:
    """Test energy budget observational constraint."""

    def test_initialization(self):
        """Test energy budget constraint initialization."""
        eb = EnergyBudgetConstraint(time_period=(1850, 2020), n_samples=1000)
        assert eb.time_period == (1850, 2020)
        assert eb.n_samples == 1000

    def test_ecs_estimation(self):
        """Test ECS estimation from energy budget."""
        eb = EnergyBudgetConstraint(n_samples=1000)

        result = eb.estimate_ecs(
            temperature_change=1.1,
            temperature_uncertainty=0.15,
            forcing_change=2.7,
            forcing_uncertainty=0.3,
            ocean_heat_uptake=0.6,
            ocean_heat_uptake_uncertainty=0.15
        )

        # Check physical reasonableness
        assert result.ecs_mean > 0
        assert result.ecs_mean < 10
        assert result.ecs_std > 0
        assert result.ecs_5th_percentile < result.ecs_mean < result.ecs_95th_percentile

    def test_historical_data_estimation(self):
        """Test estimation from default historical data."""
        eb = EnergyBudgetConstraint(n_samples=1000)
        result = eb.estimate_from_historical_data()

        # Should return reasonable ECS
        assert result.ecs_mean > 1.5
        assert result.ecs_mean < 6.0
        assert result.effective_radiative_forcing > 0


class TestEmergentConstraint:
    """Test emergent constraint implementation."""

    def test_initialization(self):
        """Test emergent constraint initialization."""
        ec = EmergentConstraint(time_period=(1980, 2020), n_samples=1000)
        assert ec.time_period == (1980, 2020)
        assert ec.n_samples == 1000

    def test_ecs_estimation(self):
        """Test ECS estimation from emergent relationship."""
        ec = EmergentConstraint(n_samples=1000, use_cross_validation=False)

        # Create synthetic model ensemble
        np.random.seed(42)
        n_models = 15
        predictor = np.random.uniform(0, 10, n_models)
        ecs_values = 2.0 + 0.3 * predictor + np.random.normal(0, 0.3, n_models)

        # Observed predictor value
        observed = 5.0
        observed_unc = 0.5

        result = ec.estimate_ecs(
            model_predictor_values=predictor,
            model_ecs_values=ecs_values,
            observed_predictor=observed,
            observed_predictor_uncertainty=observed_unc,
            predictor_name="test_predictor"
        )

        # Check results
        assert result.ecs_mean > 0
        assert result.ecs_std > 0
        assert result.metadata["correlation"] > 0  # Should be positive correlation
        assert result.metadata["n_models"] == n_models

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        ec = EmergentConstraint(n_samples=1000, use_cross_validation=True)

        # Create synthetic data with strong relationship
        np.random.seed(42)
        n_models = 20
        predictor = np.linspace(0, 10, n_models)
        ecs_values = 2.0 + 0.5 * predictor + np.random.normal(0, 0.2, n_models)

        result = ec.estimate_ecs(
            model_predictor_values=predictor,
            model_ecs_values=ecs_values,
            observed_predictor=5.0,
            observed_predictor_uncertainty=0.3
        )

        # Cross-validation score should be high for strong relationship
        assert result.metadata["cv_score"] is not None
        assert result.metadata["cv_score"] > 0.5  # Reasonable fit


class TestCloudFeedbackConstraint:
    """Test cloud feedback process-based constraint."""

    def test_initialization(self):
        """Test cloud feedback constraint initialization."""
        cfc = CloudFeedbackConstraint(n_samples=1000)
        assert cfc.n_samples == 1000

    def test_ecs_from_feedbacks(self):
        """Test ECS estimation from feedback components."""
        cfc = CloudFeedbackConstraint(n_samples=1000)

        # Use typical CMIP6 feedback values
        feedbacks = {
            "planck": (-3.2, 0.05),
            "water_vapor": (1.80, 0.15),
            "lapse_rate": (-0.80, 0.20),
            "cloud": (0.42, 0.35),
            "albedo": (0.35, 0.10)
        }

        result = cfc.estimate_ecs_from_feedbacks(feedbacks)

        # Check physical reasonableness
        assert result.ecs_mean > 0
        assert result.ecs_mean < 10
        assert result.total_feedback < 0  # Net negative feedback (stable system)

    def test_feedback_decomposition(self):
        """Test cloud feedback decomposition."""
        cfc = CloudFeedbackConstraint(n_samples=1000)

        decomposed = cfc.decompose_cloud_feedback(
            low_cloud=(0.30, 0.25),
            high_cloud=(0.10, 0.15),
            middle_cloud=(0.02, 0.10)
        )

        # Check that total is sum of components
        total_mean = decomposed["cloud_total"][0]
        sum_components = (
            decomposed["cloud_low"][0] +
            decomposed["cloud_high"][0] +
            decomposed["cloud_middle"][0]
        )

        assert np.isclose(total_mean, sum_components)


class TestConstraintCombination:
    """Test combining multiple constraints."""

    def test_multiple_constraints(self):
        """Test combining paleoclimate and observational constraints."""
        # Create two constraints
        lgm = LGMConstraint(n_samples=1000)
        eb = EnergyBudgetConstraint(n_samples=1000)

        lgm_result = lgm.estimate_ecs(-6.0, 0.4, -7.5, 1.0)
        eb_result = eb.estimate_from_historical_data()

        # Both should give reasonable ECS values
        assert lgm_result.ecs_mean > 1.0
        assert eb_result.ecs_mean > 1.0

        # Combined uncertainty should be less than individual uncertainties
        # (This would be tested in the Bayesian integration module)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
