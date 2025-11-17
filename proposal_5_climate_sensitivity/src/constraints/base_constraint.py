"""
Base Constraint Class

Abstract base class for all climate sensitivity constraints.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)


class BaseConstraint(ABC):
    """
    Abstract base class for climate sensitivity constraints.

    All specific constraint implementations should inherit from this class
    and implement the required abstract methods.

    Attributes:
        name (str): Name of the constraint
        constraint_type (str): Type of constraint (paleoclimate, observational, process_based)
        metadata (dict): Additional metadata about the constraint
    """

    def __init__(
        self,
        name: str,
        constraint_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base constraint.

        Args:
            name: Name of the constraint
            constraint_type: Type of constraint
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.constraint_type = constraint_type
        self.metadata = metadata or {}

        logger.info(f"Initialized {self.name} constraint ({self.constraint_type})")

    @abstractmethod
    def apply(
        self,
        cmip_data: xr.Dataset,
        constraint_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply the constraint to CMIP model data.

        This method must be implemented by each specific constraint.

        Args:
            cmip_data: CMIP model data
            constraint_data: Constraint-specific data (observations, proxies, etc.)
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing:
                - ecs_constrained: Constrained ECS distribution
                - weights: Model weights based on constraint
                - diagnostics: Additional diagnostic information
        """
        pass

    def evaluate_models(
        self,
        cmip_data: xr.Dataset,
        observable: np.ndarray,
        observed_value: float,
        observed_uncertainty: float
    ) -> np.ndarray:
        """
        Evaluate model performance against an observable.

        Uses Gaussian likelihood to weight models based on how well they
        match the observed value.

        Args:
            cmip_data: CMIP model data
            observable: Observable values from each model
            observed_value: Observed value from constraint
            observed_uncertainty: Uncertainty in observed value

        Returns:
            Array of weights for each model
        """
        # Calculate likelihood using Gaussian distribution
        # L(model | obs) ∝ exp(-(obs - model)² / (2σ²))

        likelihood = np.exp(
            -0.5 * ((observable - observed_value) / observed_uncertainty) ** 2
        )

        # Normalize weights
        weights = likelihood / likelihood.sum()

        return weights

    def propagate_to_ecs(
        self,
        weights: np.ndarray,
        model_ecs: np.ndarray,
        n_samples: int = 10000
    ) -> np.ndarray:
        """
        Propagate model weights to constrained ECS distribution.

        Args:
            weights: Model weights from constraint
            model_ecs: ECS values for each model
            n_samples: Number of samples for distribution

        Returns:
            Array of samples from constrained ECS distribution
        """
        # Sample models according to weights
        model_indices = np.random.choice(
            len(model_ecs),
            size=n_samples,
            p=weights / weights.sum()
        )

        # Get corresponding ECS values
        ecs_samples = model_ecs[model_indices]

        # Add within-model uncertainty (typically ~10-20%)
        within_model_uncertainty = 0.15
        ecs_samples += np.random.normal(
            0,
            within_model_uncertainty * np.abs(ecs_samples),
            n_samples
        )

        return ecs_samples

    def test_independence(
        self,
        other_constraint: 'BaseConstraint',
        cmip_data: xr.Dataset
    ) -> float:
        """
        Test independence with another constraint.

        Computes correlation between model weights from two constraints.
        High correlation suggests constraints may not be independent.

        Args:
            other_constraint: Another constraint to test independence with
            cmip_data: CMIP model data

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # This is a placeholder - actual implementation would need
        # to compute weights from both constraints and correlate them
        logger.warning(
            "Independence testing not fully implemented. "
            "Use with caution."
        )

        return 0.0

    def validate(self, perfect_model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate constraint using perfect model tests.

        In a perfect model test, we pretend one model is "truth" and
        see if the constraint can recover its properties.

        Args:
            perfect_model_results: Results from perfect model tests

        Returns:
            Validation metrics
        """
        logger.info(f"Validating {self.name} constraint")

        # Placeholder for validation logic
        validation_metrics = {
            'bias': 0.0,
            'rmse': 0.0,
            'skill_score': 0.0
        }

        return validation_metrics

    def get_summary(self) -> str:
        """
        Get a summary description of the constraint.

        Returns:
            String description of the constraint
        """
        summary = f"""
        Constraint: {self.name}
        Type: {self.constraint_type}
        """

        if self.metadata:
            summary += "\nMetadata:\n"
            for key, value in self.metadata.items():
                summary += f"  {key}: {value}\n"

        return summary

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.constraint_type}')"
