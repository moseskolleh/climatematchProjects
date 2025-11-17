"""
Last Glacial Maximum (LGM) Constraint

Constrains climate sensitivity using LGM temperature reconstructions.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional
import logging

from ..base_constraint import BaseConstraint

logger = logging.getLogger(__name__)


class LGMConstraint(BaseConstraint):
    """
    Climate sensitivity constraint from Last Glacial Maximum.

    The LGM (~21,000 years ago) provides an out-of-sample test of
    climate sensitivity with large forcing (CO2 ~190 ppm vs 280 ppm
    pre-industrial) and substantial cooling (~5°C globally).

    Method:
    1. Compare model-simulated LGM cooling to proxy reconstructions
    2. Weight models by fit to LGM temperature patterns
    3. Propagate weights to ECS distribution

    Key challenges:
    - State-dependence: LGM climate state differs from modern
    - Ice sheet configuration: LGM had large Northern Hemisphere ice sheets
    - CO2 and other greenhouse gases were different
    - Aerosols and vegetation feedbacks
    """

    def __init__(self):
        """Initialize LGM constraint."""
        super().__init__(
            name="LGM Temperature Constraint",
            constraint_type="paleoclimate",
            metadata={
                'period': 'Last Glacial Maximum',
                'age': '~21 ka',
                'co2_lgm': 190,  # ppm
                'co2_pi': 280,   # ppm
                'reference': 'Tierney et al. (2020), Annan & Hargreaves (2013)'
            }
        )

        # Proxy-based LGM cooling estimates
        self.lgm_global_cooling = -5.0  # °C (Tierney et al. 2020)
        self.lgm_cooling_uncertainty = 0.7  # °C

        # Tropical cooling (important for constraining cloud feedbacks)
        self.lgm_tropical_cooling = -3.5  # °C
        self.lgm_tropical_uncertainty = 1.0  # °C

    def apply(
        self,
        cmip_data: xr.Dataset,
        constraint_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply LGM constraint to CMIP models.

        Args:
            cmip_data: CMIP model data (should include LGM simulations or ECS values)
            constraint_data: Dictionary with proxy data (if None, uses defaults)
            **kwargs: Additional options

        Returns:
            Dictionary with constrained ECS and diagnostics
        """
        logger.info("Applying LGM temperature constraint")

        # Extract or compute model ECS values
        if 'ecs' in cmip_data.data_vars:
            model_ecs = cmip_data['ecs'].values
        elif hasattr(cmip_data, 'attrs') and 'true_ecs' in cmip_data.attrs:
            # For synthetic data
            model_ecs = np.array([
                float(cmip_data.sel(model=m).attrs.get('true_ecs', 3.0))
                for m in cmip_data.model.values
            ])
        else:
            raise ValueError("ECS values not found in CMIP data")

        # Use constraint data if provided
        if constraint_data:
            lgm_cooling = constraint_data.get(
                'global_cooling',
                self.lgm_global_cooling
            )
            lgm_uncertainty = constraint_data.get(
                'cooling_uncertainty',
                self.lgm_cooling_uncertainty
            )
        else:
            lgm_cooling = self.lgm_global_cooling
            lgm_uncertainty = self.lgm_cooling_uncertainty

        # Calculate expected LGM cooling for each model based on its ECS
        # Using energy balance: ΔT_LGM = S * ΔF_LGM / ΔF_2x
        #
        # Where:
        # S = ECS
        # ΔF_LGM = forcing from CO2 + ice sheets + vegetation
        # ΔF_2x = forcing from 2xCO2

        # CO2 forcing
        delta_f_co2_lgm = 5.35 * np.log(190 / 280)  # W/m² (negative)

        # Ice sheet forcing (albedo effect, approximate)
        delta_f_ice = -3.0  # W/m²

        # Total LGM forcing (simplified)
        delta_f_lgm = delta_f_co2_lgm + delta_f_ice  # ~-6 W/m²

        # 2xCO2 forcing
        delta_f_2x = 5.35 * np.log(2)  # ~3.7 W/m²

        # Expected cooling for each model
        # Account for state-dependence with correction factor
        state_dependence_factor = kwargs.get('state_dependence_factor', 0.9)

        expected_cooling = (
            state_dependence_factor *
            model_ecs * delta_f_lgm / delta_f_2x
        )

        # Evaluate model weights based on fit to LGM cooling
        weights = self.evaluate_models(
            cmip_data=cmip_data,
            observable=expected_cooling,
            observed_value=lgm_cooling,
            observed_uncertainty=lgm_uncertainty
        )

        # Propagate to constrained ECS distribution
        ecs_constrained = self.propagate_to_ecs(
            weights=weights,
            model_ecs=model_ecs,
            n_samples=10000
        )

        # Calculate diagnostics
        diagnostics = {
            'model_lgm_cooling': expected_cooling,
            'observed_lgm_cooling': lgm_cooling,
            'observed_lgm_uncertainty': lgm_uncertainty,
            'model_weights': weights,
            'prior_ecs_mean': float(np.mean(model_ecs)),
            'prior_ecs_std': float(np.std(model_ecs)),
            'posterior_ecs_mean': float(np.mean(ecs_constrained)),
            'posterior_ecs_std': float(np.std(ecs_constrained)),
            'posterior_ecs_5': float(np.percentile(ecs_constrained, 5)),
            'posterior_ecs_50': float(np.percentile(ecs_constrained, 50)),
            'posterior_ecs_95': float(np.percentile(ecs_constrained, 95)),
        }

        # Calculate effective number of models (inverse participation ratio)
        effective_n_models = 1.0 / np.sum(weights ** 2)
        diagnostics['effective_n_models'] = float(effective_n_models)

        logger.info(
            f"LGM constraint: Prior ECS = {diagnostics['prior_ecs_mean']:.2f} ± "
            f"{diagnostics['prior_ecs_std']:.2f} K"
        )
        logger.info(
            f"LGM constraint: Posterior ECS = {diagnostics['posterior_ecs_mean']:.2f} ± "
            f"{diagnostics['posterior_ecs_std']:.2f} K "
            f"({diagnostics['posterior_ecs_5']:.2f} - {diagnostics['posterior_ecs_95']:.2f} K)"
        )

        return {
            'ecs_constrained': ecs_constrained,
            'weights': weights,
            'diagnostics': diagnostics
        }

    def apply_with_spatial_pattern(
        self,
        cmip_data: xr.Dataset,
        proxy_data: xr.Dataset
    ) -> Dict[str, Any]:
        """
        Apply constraint using spatial pattern of LGM cooling.

        This more sophisticated version compares the spatial pattern
        of model-simulated cooling to proxy reconstructions.

        Args:
            cmip_data: CMIP model data with LGM simulations
            proxy_data: Gridded or point proxy temperature data

        Returns:
            Dictionary with constrained ECS and diagnostics
        """
        logger.info("Applying LGM constraint with spatial patterns")

        # This would implement spatial pattern matching
        # For now, fall back to global mean method
        logger.warning(
            "Spatial pattern matching not fully implemented. "
            "Using global mean method."
        )

        return self.apply(cmip_data, None)

    def account_for_state_dependence(
        self,
        ecs_lgm: float,
        temperature_state: float = -5.0
    ) -> float:
        """
        Account for potential state-dependence of climate sensitivity.

        Climate sensitivity may vary with the climate state. This method
        provides a correction factor.

        Args:
            ecs_lgm: Effective climate sensitivity inferred from LGM
            temperature_state: Temperature difference from pre-industrial

        Returns:
            Corrected ECS for modern/future climate
        """
        # Simplified correction based on literature
        # Some evidence suggests sensitivity may be slightly lower in cold climates

        # Linear correction (very simplified)
        correction_factor = 1.0 + 0.02 * temperature_state  # ~10% correction for LGM

        ecs_corrected = ecs_lgm / correction_factor

        logger.debug(
            f"State-dependence correction: {ecs_lgm:.2f} K -> {ecs_corrected:.2f} K"
        )

        return ecs_corrected
