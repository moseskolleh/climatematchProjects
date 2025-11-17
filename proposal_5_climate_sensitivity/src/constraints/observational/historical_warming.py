"""
Historical Warming Constraint

Constrains climate sensitivity using observed historical warming trend.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional
import logging

from ..base_constraint import BaseConstraint

logger = logging.getLogger(__name__)


class HistoricalWarmingConstraint(BaseConstraint):
    """
    Constraint from historical warming (1850-present).

    Uses the observed warming trend and estimated forcings to constrain ECS.

    Key challenges:
    - Aerosol forcing uncertainty
    - Internal climate variability
    - Ocean heat uptake rate
    - Incomplete equilibration
    """

    def __init__(self):
        """Initialize historical warming constraint."""
        super().__init__(
            name="Historical Warming Constraint",
            constraint_type="observational",
            metadata={
                'period': '1850-2020',
                'observed_warming': 1.1,  # °C
                'reference': 'Sherwood et al. (2020)'
            }
        )

        self.observed_warming = 1.1  # °C (1850-1900 to 2010-2020)
        self.warming_uncertainty = 0.1  # °C

        # Forcing estimates (W/m²)
        self.total_forcing = 2.7  # Net anthropogenic forcing
        self.forcing_uncertainty = 0.5  # W/m²

    def apply(
        self,
        cmip_data: xr.Dataset,
        constraint_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply historical warming constraint.

        Args:
            cmip_data: CMIP model data
            constraint_data: Observational data
            **kwargs: Additional options

        Returns:
            Dictionary with constrained ECS
        """
        logger.info("Applying historical warming constraint")

        # Extract model ECS
        if hasattr(cmip_data, 'attrs') and 'true_ecs' in cmip_data.attrs:
            model_ecs = np.array([
                float(cmip_data.sel(model=m).attrs.get('true_ecs', 3.0))
                for m in cmip_data.model.values
            ])
        else:
            model_ecs = np.array([3.0] * len(cmip_data.model))

        # Calculate expected transient warming
        # Accounts for ocean heat uptake (realized warming fraction ~0.6-0.7)
        realized_fraction = 0.65
        delta_f_2x = 3.7  # W/m²

        expected_warming = (
            realized_fraction * model_ecs * self.total_forcing / delta_f_2x
        )

        # Evaluate weights
        weights = self.evaluate_models(
            cmip_data=cmip_data,
            observable=expected_warming,
            observed_value=self.observed_warming,
            observed_uncertainty=np.sqrt(
                self.warming_uncertainty**2 +
                (realized_fraction * model_ecs * self.forcing_uncertainty / delta_f_2x)**2
            )
        )

        # Propagate to ECS
        ecs_constrained = self.propagate_to_ecs(weights, model_ecs)

        diagnostics = {
            'posterior_ecs_mean': float(np.mean(ecs_constrained)),
            'posterior_ecs_std': float(np.std(ecs_constrained)),
        }

        return {
            'ecs_constrained': ecs_constrained,
            'weights': weights,
            'diagnostics': diagnostics
        }
