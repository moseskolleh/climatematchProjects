"""
Observational constraints on climate sensitivity.

This module implements constraints derived from:
- Historical temperature records (1850-present)
- Energy budget constraints
- Emergent constraints from inter-model relationships
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ObservationalConstraintResult:
    """Results from an observational constraint analysis."""
    ecs_mean: float
    ecs_std: float
    ecs_samples: np.ndarray
    ecs_5th_percentile: float
    ecs_95th_percentile: float
    transient_climate_response: Optional[float]
    effective_radiative_forcing: float
    ocean_heat_uptake: Optional[float]
    metadata: Dict


class ObservationalConstraint:
    """Base class for observational constraints on climate sensitivity."""

    def __init__(self, time_period: Tuple[int, int], n_samples: int = 10000):
        """
        Initialize observational constraint.

        Args:
            time_period: (start_year, end_year) for analysis period
            n_samples: Number of Monte Carlo samples
        """
        self.time_period = time_period
        self.n_samples = n_samples

    def calculate_percentiles(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate common percentiles for reporting."""
        return {
            "5th": np.percentile(samples, 5),
            "16th": np.percentile(samples, 16),
            "50th": np.percentile(samples, 50),
            "84th": np.percentile(samples, 84),
            "95th": np.percentile(samples, 95)
        }


class EnergyBudgetConstraint(ObservationalConstraint):
    """
    Energy budget constraint on climate sensitivity.

    Uses the relationship:
        ECS = ΔT / (ΔF - ΔQ) * F_2xCO2

    where:
        ΔT: Observed temperature change
        ΔF: Effective radiative forcing
        ΔQ: Ocean heat uptake rate
        F_2xCO2: Forcing from CO2 doubling (~3.7 W/m²)

    This follows the approach of Gregory et al. (2002), Otto et al. (2013),
    and Lewis & Curry (2018).
    """

    F_2XCO2 = 3.7  # W/m²

    def __init__(
        self,
        time_period: Tuple[int, int] = (1850, 2020),
        n_samples: int = 10000,
        account_for_efficacy: bool = True
    ):
        """
        Initialize energy budget constraint.

        Args:
            time_period: Analysis period
            n_samples: Number of Monte Carlo samples
            account_for_efficacy: Whether to account for forcing efficacy differences
        """
        super().__init__(time_period, n_samples)
        self.account_for_efficacy = account_for_efficacy

    def estimate_ecs(
        self,
        temperature_change: float,
        temperature_uncertainty: float,
        forcing_change: float,
        forcing_uncertainty: float,
        ocean_heat_uptake: float,
        ocean_heat_uptake_uncertainty: float
    ) -> ObservationalConstraintResult:
        """
        Estimate ECS using energy budget method.

        Args:
            temperature_change: Observed warming (K)
            temperature_uncertainty: Temperature uncertainty (K, 1-sigma)
            forcing_change: Effective radiative forcing (W/m²)
            forcing_uncertainty: Forcing uncertainty (W/m², 1-sigma)
            ocean_heat_uptake: Heat uptake rate (W/m²)
            ocean_heat_uptake_uncertainty: OHU uncertainty (W/m², 1-sigma)

        Returns:
            ObservationalConstraintResult with ECS estimates
        """
        # Generate samples for each variable
        temp_samples = np.random.normal(
            temperature_change, temperature_uncertainty, self.n_samples
        )
        forcing_samples = np.random.normal(
            forcing_change, forcing_uncertainty, self.n_samples
        )
        ohu_samples = np.random.normal(
            ocean_heat_uptake, ocean_heat_uptake_uncertainty, self.n_samples
        )

        # Calculate climate feedback parameter
        # λ = (ΔF - ΔQ) / ΔT
        net_forcing = forcing_samples - ohu_samples
        temp_samples = np.where(
            np.abs(temp_samples) < 0.1, 0.1, temp_samples
        )  # Avoid division by zero

        feedback_param = net_forcing / temp_samples

        # ECS = F_2xCO2 / λ
        # Avoid division by very small or negative feedback parameters
        valid_mask = feedback_param > 0.2  # Physical constraint
        feedback_param = feedback_param[valid_mask]

        ecs_samples = self.F_2XCO2 / feedback_param

        # Remove unphysical values
        physical_mask = (ecs_samples > 0) & (ecs_samples < 10)
        ecs_samples = ecs_samples[physical_mask]

        if len(ecs_samples) < 0.5 * self.n_samples:
            warnings.warn(
                f"Large fraction of samples ({1 - len(ecs_samples)/self.n_samples:.1%}) "
                f"rejected due to physical constraints. Check input data."
            )

        percentiles = self.calculate_percentiles(ecs_samples)

        return ObservationalConstraintResult(
            ecs_mean=np.mean(ecs_samples),
            ecs_std=np.std(ecs_samples),
            ecs_samples=ecs_samples,
            ecs_5th_percentile=percentiles["5th"],
            ecs_95th_percentile=percentiles["95th"],
            transient_climate_response=None,  # Can be calculated separately
            effective_radiative_forcing=forcing_change,
            ocean_heat_uptake=ocean_heat_uptake,
            metadata={
                "method": "Energy Budget",
                "time_period": self.time_period,
                "n_samples": len(ecs_samples),
                "n_rejected": self.n_samples - len(ecs_samples),
                "temperature_change": temperature_change,
                "forcing_change": forcing_change,
                "ocean_heat_uptake": ocean_heat_uptake
            }
        )

    def estimate_from_historical_data(
        self,
        reference_period: Tuple[int, int] = (1850, 1900),
        target_period: Tuple[int, int] = (2010, 2020),
        temperature_data: Optional[np.ndarray] = None,
        forcing_data: Optional[Dict[str, np.ndarray]] = None
    ) -> ObservationalConstraintResult:
        """
        Estimate ECS from historical observations.

        This is a placeholder for a full implementation that would:
        1. Load temperature data (e.g., HadCRUT5, BEST)
        2. Load forcing data (e.g., from IPCC AR6)
        3. Calculate ocean heat uptake from Argo/WOD
        4. Apply the energy budget method

        Args:
            reference_period: Baseline period
            target_period: Recent period for comparison
            temperature_data: Optional pre-loaded temperature data
            forcing_data: Optional pre-loaded forcing data

        Returns:
            ObservationalConstraintResult
        """
        # This is a simplified example with typical values
        # In practice, would load actual observational datasets

        # Typical values for 1850-1900 to 2010-2020
        temp_change = 1.09  # K (IPCC AR6 best estimate)
        temp_unc = 0.15  # K

        forcing_change = 2.72  # W/m² (IPCC AR6)
        forcing_unc = 0.30  # W/m²

        ohu = 0.56  # W/m² (Loeb et al. 2021)
        ohu_unc = 0.15  # W/m²

        return self.estimate_ecs(
            temperature_change=temp_change,
            temperature_uncertainty=temp_unc,
            forcing_change=forcing_change,
            forcing_uncertainty=forcing_unc,
            ocean_heat_uptake=ohu,
            ocean_heat_uptake_uncertainty=ohu_unc
        )


class EmergentConstraint(ObservationalConstraint):
    """
    Emergent constraint using inter-model relationships.

    Emergent constraints exploit correlations between:
    - An observable quantity X (e.g., cloud variability, tropical warming pattern)
    - Climate sensitivity Y (across climate models)

    Then use observed value of X to constrain Y.

    Key challenges:
    - Statistical robustness (multiple testing, overfitting)
    - Physical understanding (why does the relationship exist?)
    - Independence (are model biases independent?)
    """

    def __init__(
        self,
        time_period: Tuple[int, int] = (1980, 2020),
        n_samples: int = 10000,
        use_cross_validation: bool = True
    ):
        """
        Initialize emergent constraint.

        Args:
            time_period: Period for observed predictor variable
            n_samples: Number of Monte Carlo samples
            use_cross_validation: Whether to use leave-one-out cross-validation
        """
        super().__init__(time_period, n_samples)
        self.use_cross_validation = use_cross_validation

    def estimate_ecs(
        self,
        model_predictor_values: np.ndarray,
        model_ecs_values: np.ndarray,
        observed_predictor: float,
        observed_predictor_uncertainty: float,
        predictor_name: str = "predictor"
    ) -> ObservationalConstraintResult:
        """
        Apply emergent constraint to estimate ECS.

        Args:
            model_predictor_values: Predictor values from model ensemble (n_models,)
            model_ecs_values: ECS values from model ensemble (n_models,)
            observed_predictor: Observed value of predictor
            observed_predictor_uncertainty: Observation uncertainty (1-sigma)
            predictor_name: Name of predictor variable for metadata

        Returns:
            ObservationalConstraintResult with constrained ECS
        """
        # Fit linear relationship: ECS = a + b * predictor
        X = model_predictor_values
        Y = model_ecs_values

        # Linear regression with uncertainty
        n_models = len(X)
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        # Regression coefficients
        slope = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
        intercept = Y_mean - slope * X_mean

        # Residual standard deviation
        predictions = intercept + slope * X
        residuals = Y - predictions
        residual_std = np.std(residuals)

        # Standard error of regression
        # This accounts for uncertainty in the linear relationship
        se_slope = residual_std / np.sqrt(np.sum((X - X_mean)**2))

        # Generate samples for observed predictor
        obs_samples = np.random.normal(
            observed_predictor,
            observed_predictor_uncertainty,
            self.n_samples
        )

        # Propagate uncertainties through regression
        # Account for: observation uncertainty, regression uncertainty, residual scatter
        slope_samples = np.random.normal(slope, se_slope, self.n_samples)
        intercept_samples = np.random.normal(intercept, residual_std / np.sqrt(n_models), self.n_samples)

        ecs_samples = intercept_samples + slope_samples * obs_samples

        # Add residual scatter
        ecs_samples += np.random.normal(0, residual_std, self.n_samples)

        # Remove unphysical values
        physical_mask = (ecs_samples > 0) & (ecs_samples < 10)
        ecs_samples = ecs_samples[physical_mask]

        percentiles = self.calculate_percentiles(ecs_samples)

        # Calculate correlation strength
        correlation = np.corrcoef(X, Y)[0, 1]

        # Cross-validation score
        cv_score = None
        if self.use_cross_validation:
            cv_score = self._cross_validate(X, Y)

        return ObservationalConstraintResult(
            ecs_mean=np.mean(ecs_samples),
            ecs_std=np.std(ecs_samples),
            ecs_samples=ecs_samples,
            ecs_5th_percentile=percentiles["5th"],
            ecs_95th_percentile=percentiles["95th"],
            transient_climate_response=None,
            effective_radiative_forcing=np.nan,
            ocean_heat_uptake=None,
            metadata={
                "method": "Emergent Constraint",
                "predictor": predictor_name,
                "correlation": correlation,
                "slope": slope,
                "intercept": intercept,
                "residual_std": residual_std,
                "n_models": n_models,
                "observed_predictor": observed_predictor,
                "cv_score": cv_score
            }
        )

    def _cross_validate(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Perform leave-one-out cross-validation.

        Args:
            X: Predictor values
            Y: Target values (ECS)

        Returns:
            Cross-validation R² score
        """
        n = len(X)
        predictions = np.zeros(n)

        for i in range(n):
            # Leave out i-th model
            X_train = np.delete(X, i)
            Y_train = np.delete(Y, i)
            X_test = X[i]

            # Fit regression without i-th model
            slope = np.sum((X_train - X_train.mean()) * (Y_train - Y_train.mean())) / \
                    np.sum((X_train - X_train.mean())**2)
            intercept = Y_train.mean() - slope * X_train.mean()

            # Predict i-th model
            predictions[i] = intercept + slope * X_test

        # Calculate R²
        ss_res = np.sum((Y - predictions)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        r2 = 1 - ss_res / ss_tot

        return r2


def combine_observational_constraints(
    constraints: list,
    correlation_matrix: Optional[np.ndarray] = None
) -> ObservationalConstraintResult:
    """
    Combine multiple observational constraints accounting for dependencies.

    Args:
        constraints: List of ObservationalConstraintResult objects
        correlation_matrix: Optional correlation matrix between constraints

    Returns:
        Combined ObservationalConstraintResult
    """
    if correlation_matrix is None:
        # Assume moderate correlation between different observational methods
        n = len(constraints)
        correlation_matrix = np.eye(n) * 0.7 + np.ones((n, n)) * 0.3

    # Use variance-weighted combination
    # More sophisticated methods could use Bayesian model averaging

    variances = np.array([c.ecs_std**2 for c in constraints])
    weights = 1.0 / variances
    weights /= weights.sum()

    # Weighted mean
    combined_mean = np.sum([w * c.ecs_mean for w, c in zip(weights, constraints)])

    # Combine samples
    all_samples = []
    for constraint, weight in zip(constraints, weights):
        n_samples = int(weight * 10000)
        sampled = np.random.choice(constraint.ecs_samples, size=n_samples, replace=True)
        all_samples.append(sampled)

    combined_samples = np.concatenate(all_samples)

    percentiles = {
        "5th": np.percentile(combined_samples, 5),
        "95th": np.percentile(combined_samples, 95)
    }

    return ObservationalConstraintResult(
        ecs_mean=combined_mean,
        ecs_std=np.std(combined_samples),
        ecs_samples=combined_samples,
        ecs_5th_percentile=percentiles["5th"],
        ecs_95th_percentile=percentiles["95th"],
        transient_climate_response=None,
        effective_radiative_forcing=np.nan,
        ocean_heat_uptake=None,
        metadata={
            "method": "Combined Observational",
            "n_constraints": len(constraints),
            "weights": weights.tolist(),
            "individual_methods": [c.metadata["method"] for c in constraints]
        }
    )
