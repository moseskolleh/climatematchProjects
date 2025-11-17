"""
Energy Budget Constraint

Constrains ECS using Earth's energy imbalance and ocean heat uptake.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional
from ..base_constraint import BaseConstraint
import logging

logger = logging.getLogger(__name__)


class EnergyBudgetConstraint(BaseConstraint):
    """Constraint from energy budget analysis."""

    def __init__(self):
        super().__init__(
            name="Energy Budget Constraint",
            constraint_type="observational",
            metadata={'reference': 'CERES + Argo data'}
        )

        self.toa_imbalance = 0.7  # W/m² (current Earth energy imbalance)
        self.imbalance_uncertainty = 0.2  # W/m²

    def apply(
        self,
        cmip_data: xr.Dataset,
        constraint_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply energy budget constraint."""
        logger.info("Applying energy budget constraint")

        # Extract ECS
        if hasattr(cmip_data, 'attrs') and 'true_ecs' in cmip_data.attrs:
            model_ecs = np.array([
                float(cmip_data.sel(model=m).attrs.get('true_ecs', 3.0))
                for m in cmip_data.model.values
            ])
        else:
            model_ecs = np.array([3.0] * len(cmip_data.model))

        # Simple model: N = F - lambda * ΔT
        # where lambda = F_2x / ECS
        forcing = 2.7  # Current total forcing (W/m²)
        delta_t = 1.1  # Current warming (K)
        delta_f_2x = 3.7  # 2xCO2 forcing (W/m²)

        lambda_param = delta_f_2x / model_ecs
        expected_imbalance = forcing - lambda_param * delta_t

        weights = self.evaluate_models(
            cmip_data=cmip_data,
            observable=expected_imbalance,
            observed_value=self.toa_imbalance,
            observed_uncertainty=self.imbalance_uncertainty
        )

        ecs_constrained = self.propagate_to_ecs(weights, model_ecs)

        return {
            'ecs_constrained': ecs_constrained,
            'weights': weights,
            'diagnostics': {
                'posterior_ecs_mean': float(np.mean(ecs_constrained)),
                'posterior_ecs_std': float(np.std(ecs_constrained)),
            }
        }
