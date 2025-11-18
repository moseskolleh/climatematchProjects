"""
Paleoclimate constraints on climate sensitivity.

This module implements constraints derived from paleoclimate records including:
- Last Glacial Maximum (LGM, ~21 ka)
- Mid-Pliocene Warm Period (mPWP, ~3 Ma)
- Last Interglacial (~130-115 ka)

The basic approach is:
    ECS = ΔT_paleo / (ΔF_paleo / F_2xCO2)
where F_2xCO2 ≈ 3.7 W/m²
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class PaleoConstraintResult:
    """Results from a paleoclimate constraint analysis."""
    ecs_mean: float
    ecs_std: float
    ecs_samples: np.ndarray
    temperature_change: float
    temperature_uncertainty: float
    forcing_change: float
    forcing_uncertainty: float
    state_dependence_factor: float
    metadata: Dict


class PaleoclimateConstraint:
    """
    Base class for paleoclimate-based climate sensitivity constraints.

    Attributes:
        F_2xCO2 (float): Radiative forcing from CO2 doubling (W/m²)
        reference_period (str): Name of the paleoclimate period
    """

    F_2XCO2 = 3.7  # W/m² radiative forcing from CO2 doubling

    def __init__(
        self,
        reference_period: str,
        apply_state_dependence: bool = True,
        n_samples: int = 10000
    ):
        """
        Initialize paleoclimate constraint.

        Args:
            reference_period: Name of the paleoclimate period
            apply_state_dependence: Whether to apply state-dependence corrections
            n_samples: Number of Monte Carlo samples for uncertainty propagation
        """
        self.reference_period = reference_period
        self.apply_state_dependence = apply_state_dependence
        self.n_samples = n_samples

    def estimate_ecs(
        self,
        temperature_change: float,
        temperature_uncertainty: float,
        forcing_change: float,
        forcing_uncertainty: float,
        correlation: float = 0.0
    ) -> PaleoConstraintResult:
        """
        Estimate ECS from paleoclimate temperature and forcing changes.

        Args:
            temperature_change: Global mean temperature change (K)
            temperature_uncertainty: Uncertainty in temperature (K, 1-sigma)
            forcing_change: Effective radiative forcing change (W/m²)
            forcing_uncertainty: Uncertainty in forcing (W/m², 1-sigma)
            correlation: Correlation between temperature and forcing uncertainties

        Returns:
            PaleoConstraintResult with ECS estimates and uncertainties
        """
        # Generate correlated samples for temperature and forcing
        mean = [temperature_change, forcing_change]
        cov = [
            [temperature_uncertainty**2,
             correlation * temperature_uncertainty * forcing_uncertainty],
            [correlation * temperature_uncertainty * forcing_uncertainty,
             forcing_uncertainty**2]
        ]

        samples = np.random.multivariate_normal(mean, cov, size=self.n_samples)
        temp_samples = samples[:, 0]
        forcing_samples = samples[:, 1]

        # Calculate ECS samples: ECS = ΔT / (ΔF / F_2xCO2)
        # Avoid division by zero
        forcing_samples = np.where(
            np.abs(forcing_samples) < 0.1,
            np.sign(forcing_samples) * 0.1,
            forcing_samples
        )

        ecs_samples = temp_samples / (forcing_samples / self.F_2XCO2)

        # Apply state-dependence correction if requested
        state_factor = 1.0
        if self.apply_state_dependence:
            state_factor = self._state_dependence_correction(
                temperature_change, forcing_change
            )
            ecs_samples *= state_factor

        # Remove unphysical values (negative or extremely large)
        valid_mask = (ecs_samples > 0) & (ecs_samples < 15)
        if np.sum(valid_mask) < 0.5 * self.n_samples:
            warnings.warn(
                f"More than 50% of samples outside physical range [0, 15]K. "
                f"Check input data quality."
            )

        ecs_samples = ecs_samples[valid_mask]

        return PaleoConstraintResult(
            ecs_mean=np.mean(ecs_samples),
            ecs_std=np.std(ecs_samples),
            ecs_samples=ecs_samples,
            temperature_change=temperature_change,
            temperature_uncertainty=temperature_uncertainty,
            forcing_change=forcing_change,
            forcing_uncertainty=forcing_uncertainty,
            state_dependence_factor=state_factor,
            metadata={
                "reference_period": self.reference_period,
                "n_samples": len(ecs_samples),
                "n_rejected": self.n_samples - len(ecs_samples),
                "state_dependence_applied": self.apply_state_dependence
            }
        )

    def _state_dependence_correction(
        self,
        temperature_change: float,
        forcing_change: float
    ) -> float:
        """
        Calculate state-dependence correction factor.

        Climate sensitivity may differ between cold (ice age) and warm climates.
        This is a placeholder for more sophisticated corrections.

        Args:
            temperature_change: Temperature change from reference period (K)
            forcing_change: Forcing change from reference period (W/m²)

        Returns:
            Correction factor (typically 0.8 - 1.2)
        """
        # Default: no correction
        # Subclasses should override this with period-specific corrections
        return 1.0


class LGMConstraint(PaleoclimateConstraint):
    """
    Last Glacial Maximum (LGM) climate sensitivity constraint.

    The LGM (~21,000 years ago) provides a constraint from a colder climate state
    with extensive ice sheets. Key challenges:
    - State-dependence: Sensitivity may differ in cold vs warm climates
    - Ice sheet forcing: Albedo and topographic effects are uncertain
    - Proxy uncertainties: Spatial coverage limitations
    """

    def __init__(self, apply_state_dependence: bool = True, n_samples: int = 10000):
        super().__init__(
            reference_period="Last Glacial Maximum",
            apply_state_dependence=apply_state_dependence,
            n_samples=n_samples
        )

    def _state_dependence_correction(
        self,
        temperature_change: float,
        forcing_change: float
    ) -> float:
        """
        LGM-specific state-dependence correction.

        Studies suggest ECS from LGM may be ~10-20% lower than future warming
        due to different cloud feedbacks and ice-albedo dynamics.

        Based on: Tierney et al. (2020) Nature, Sherwood et al. (2020) Rev. Geophys.
        """
        # LGM was ~5-7K colder than pre-industrial
        # Apply a modest correction: cold climate sensitivity ~90% of warm climate
        correction_factor = 1.0 / 0.9  # Adjust LGM estimate upward for future

        return correction_factor

    def estimate_from_pmip4(
        self,
        pmip4_model_data: Dict[str, Tuple[float, float]],
        proxy_temperature: float,
        proxy_temperature_uncertainty: float
    ) -> PaleoConstraintResult:
        """
        Estimate ECS using PMIP4 model ensemble and proxy data.

        Args:
            pmip4_model_data: Dict of {model_name: (forcing, forcing_unc)}
            proxy_temperature: LGM cooling from proxy synthesis (K)
            proxy_temperature_uncertainty: Uncertainty in proxy temperature (K)

        Returns:
            PaleoConstraintResult with ECS estimates
        """
        # Average forcing from PMIP4 models
        forcings = np.array([f[0] for f in pmip4_model_data.values()])
        forcing_uncs = np.array([f[1] for f in pmip4_model_data.values()])

        mean_forcing = np.mean(forcings)
        # Combine model spread and individual uncertainties
        forcing_uncertainty = np.sqrt(
            np.std(forcings)**2 + np.mean(forcing_uncs)**2
        )

        return self.estimate_ecs(
            temperature_change=proxy_temperature,
            temperature_uncertainty=proxy_temperature_uncertainty,
            forcing_change=mean_forcing,
            forcing_uncertainty=forcing_uncertainty,
            correlation=-0.2  # Weak negative correlation expected
        )


class PlioceneConstraint(PaleoclimateConstraint):
    """
    Mid-Pliocene Warm Period (mPWP) climate sensitivity constraint.

    The mPWP (~3.3-3.0 Ma) was 2-4K warmer than pre-industrial with CO2
    levels similar to today (~400 ppm). Advantages:
    - Warmer climate state more analogous to future
    - No large ice sheets to complicate forcing

    Challenges:
    - Dating uncertainties in proxies
    - Orbital configuration different from today
    - Vegetation and topography differences
    """

    def __init__(self, apply_state_dependence: bool = True, n_samples: int = 10000):
        super().__init__(
            reference_period="Mid-Pliocene Warm Period",
            apply_state_dependence=apply_state_dependence,
            n_samples=n_samples
        )

    def _state_dependence_correction(
        self,
        temperature_change: float,
        forcing_change: float
    ) -> float:
        """
        Pliocene-specific state-dependence correction.

        The Pliocene is thought to be a better analogue for future warming,
        so less correction needed.
        """
        # Pliocene climate state closer to future, minimal correction
        correction_factor = 1.0
        return correction_factor

    def estimate_from_pliomip(
        self,
        pliomip_temperature: float,
        pliomip_temperature_unc: float,
        co2_pliocene: float,
        co2_preindustrial: float = 280.0,
        other_forcings: float = 0.0,
        other_forcings_unc: float = 0.5
    ) -> PaleoConstraintResult:
        """
        Estimate ECS from PlioMIP simulations.

        Args:
            pliomip_temperature: Pliocene warming relative to PI (K)
            pliomip_temperature_unc: Uncertainty in temperature (K)
            co2_pliocene: Pliocene CO2 concentration (ppm)
            co2_preindustrial: Pre-industrial CO2 (ppm)
            other_forcings: Non-CO2 forcings (ice, vegetation) (W/m²)
            other_forcings_unc: Uncertainty in other forcings (W/m²)

        Returns:
            PaleoConstraintResult with ECS estimates
        """
        # Calculate CO2 forcing using logarithmic relationship
        # F = 5.35 * ln(C/C0) W/m²
        co2_forcing = 5.35 * np.log(co2_pliocene / co2_preindustrial)

        # Total forcing includes CO2 and other factors
        total_forcing = co2_forcing + other_forcings

        # Uncertainty in CO2 forcing (assume ±20% uncertainty in formula)
        co2_forcing_unc = 0.2 * abs(co2_forcing)
        total_forcing_unc = np.sqrt(co2_forcing_unc**2 + other_forcings_unc**2)

        return self.estimate_ecs(
            temperature_change=pliomip_temperature,
            temperature_uncertainty=pliomip_temperature_unc,
            forcing_change=total_forcing,
            forcing_uncertainty=total_forcing_unc,
            correlation=0.0
        )


class LastInterglacialConstraint(PaleoclimateConstraint):
    """
    Last Interglacial (LIG) climate sensitivity constraint.

    The LIG (~130-115 ka) was ~0.5-2K warmer than pre-industrial due to
    orbital forcing. Advantages:
    - Recent enough for good proxy coverage
    - Similar climate state to present

    Challenges:
    - Primarily orbital forcing (different spatial pattern than CO2)
    - Regional vs global temperature interpretations
    - Sea level contributions uncertain
    """

    def __init__(self, apply_state_dependence: bool = True, n_samples: int = 10000):
        super().__init__(
            reference_period="Last Interglacial",
            apply_state_dependence=apply_state_dependence,
            n_samples=n_samples
        )

    def _state_dependence_correction(
        self,
        temperature_change: float,
        forcing_change: float
    ) -> float:
        """
        LIG-specific state-dependence correction.

        Orbital forcing has different spatial pattern than CO2, which may
        affect feedback strengths.
        """
        # Account for different spatial pattern of orbital vs CO2 forcing
        # Modest correction needed
        correction_factor = 1.05
        return correction_factor


def combine_paleoclimate_constraints(
    constraints: list,
    weights: Optional[np.ndarray] = None
) -> PaleoConstraintResult:
    """
    Combine multiple paleoclimate constraints using Bayesian model averaging.

    Args:
        constraints: List of PaleoConstraintResult objects
        weights: Optional weights for each constraint (default: equal weights)

    Returns:
        Combined PaleoConstraintResult
    """
    if weights is None:
        weights = np.ones(len(constraints)) / len(constraints)

    weights = np.array(weights)
    weights /= weights.sum()  # Normalize

    # Combine samples from all constraints
    all_samples = []
    for constraint, weight in zip(constraints, weights):
        # Resample according to weight
        n_samples = int(weight * 10000)
        sampled = np.random.choice(constraint.ecs_samples, size=n_samples, replace=True)
        all_samples.append(sampled)

    combined_samples = np.concatenate(all_samples)

    return PaleoConstraintResult(
        ecs_mean=np.mean(combined_samples),
        ecs_std=np.std(combined_samples),
        ecs_samples=combined_samples,
        temperature_change=np.nan,  # Not applicable for combined constraint
        temperature_uncertainty=np.nan,
        forcing_change=np.nan,
        forcing_uncertainty=np.nan,
        state_dependence_factor=np.nan,
        metadata={
            "reference_period": "Combined Paleoclimate",
            "n_constraints": len(constraints),
            "weights": weights.tolist(),
            "individual_periods": [c.metadata["reference_period"] for c in constraints]
        }
    )
