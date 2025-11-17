"""
Mid-Pliocene Warm Period (MPWP) Constraint

Constrains climate sensitivity using MPWP temperature reconstructions.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional
import logging

from ..base_constraint import BaseConstraint

logger = logging.getLogger(__name__)


class MPWPConstraint(BaseConstraint):
    """
    Climate sensitivity constraint from Mid-Pliocene Warm Period.

    The MPWP (~3.3-3.0 Ma) had CO2 levels (~400 ppm) similar to today
    but warmer temperatures, providing a near-analogue for future warming.

    Advantages:
    - Similar CO2 to modern (less state-dependence concern)
    - Good spatial proxy coverage
    - No major ice sheet differences (Greenland smaller, West Antarctic uncertain)

    Challenges:
    - Orbital configuration different
    - Some ice sheet uncertainty
    - Vegetation distribution different
    """

    def __init__(self):
        """Initialize MPWP constraint."""
        super().__init__(
            name="MPWP Temperature Constraint",
            constraint_type="paleoclimate",
            metadata={
                'period': 'Mid-Pliocene Warm Period',
                'age': '~3.3-3.0 Ma',
                'co2_mpwp': 400,  # ppm
                'co2_pi': 280,   # ppm
                'reference': 'Haywood et al. (2020), Burke et al. (2018)'
            }
        )

        # MPWP warming estimates
        self.mpwp_global_warming = 2.5  # °C
        self.mpwp_warming_uncertainty = 0.5  # °C

    def apply(
        self,
        cmip_data: xr.Dataset,
        constraint_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply MPWP constraint to CMIP models.

        Args:
            cmip_data: CMIP model data
            constraint_data: Dictionary with proxy data (if None, uses defaults)
            **kwargs: Additional options

        Returns:
            Dictionary with constrained ECS and diagnostics
        """
        logger.info("Applying MPWP temperature constraint")

        # Extract model ECS
        if hasattr(cmip_data, 'attrs') and 'true_ecs' in cmip_data.attrs:
            model_ecs = np.array([
                float(cmip_data.sel(model=m).attrs.get('true_ecs', 3.0))
                for m in cmip_data.model.values
            ])
        else:
            logger.warning("Using default ECS values")
            model_ecs = np.array([3.0] * len(cmip_data.model))

        # Calculate expected MPWP warming
        delta_f_co2 = 5.35 * np.log(400 / 280)  # W/m²
        delta_f_2x = 5.35 * np.log(2)  # W/m²

        expected_warming = model_ecs * delta_f_co2 / delta_f_2x

        # Evaluate weights
        weights = self.evaluate_models(
            cmip_data=cmip_data,
            observable=expected_warming,
            observed_value=self.mpwp_global_warming,
            observed_uncertainty=self.mpwp_warming_uncertainty
        )

        # Propagate to ECS
        ecs_constrained = self.propagate_to_ecs(weights, model_ecs)

        diagnostics = {
            'posterior_ecs_mean': float(np.mean(ecs_constrained)),
            'posterior_ecs_std': float(np.std(ecs_constrained)),
            'posterior_ecs_5_95': [
                float(np.percentile(ecs_constrained, 5)),
                float(np.percentile(ecs_constrained, 95))
            ]
        }

        return {
            'ecs_constrained': ecs_constrained,
            'weights': weights,
            'diagnostics': diagnostics
        }
