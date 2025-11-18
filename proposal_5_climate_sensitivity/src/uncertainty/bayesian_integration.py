"""
Bayesian integration of multiple climate sensitivity constraints.

This module implements hierarchical Bayesian methods to combine different
lines of evidence for climate sensitivity while accounting for dependencies
and structural uncertainties.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class MultiConstraintPosterior:
    """
    Posterior distribution from multi-constraint analysis.
    """
    ecs_samples: np.ndarray
    ecs_mean: float
    ecs_median: float
    ecs_std: float
    ecs_percentiles: Dict[str, float]
    constraint_weights: Dict[str, float]
    information_gain: float
    evidence: float
    metadata: Dict


class BayesianIntegrator:
    """
    Bayesian integration of multiple climate sensitivity constraints.

    Uses a hierarchical Bayesian framework:
        p(ECS | D₁, D₂, ..., Dₙ) ∝ p(ECS) ∏ᵢ p(Dᵢ | ECS, θᵢ) p(θᵢ)

    where Dᵢ are different constraint datasets and θᵢ are nuisance parameters.
    """

    def __init__(
        self,
        prior_type: str = "jeffreys",
        prior_params: Optional[Dict] = None,
        n_samples: int = 50000,
        burn_in: int = 10000
    ):
        """
        Initialize Bayesian integrator.

        Args:
            prior_type: Type of prior ("uniform", "jeffreys", "informed")
            prior_params: Parameters for prior distribution
            n_samples: Number of posterior samples
            burn_in: Number of burn-in samples to discard
        """
        self.prior_type = prior_type
        self.prior_params = prior_params or {}
        self.n_samples = n_samples
        self.burn_in = burn_in

    def integrate_constraints(
        self,
        constraint_data: Dict[str, np.ndarray],
        constraint_weights: Optional[Dict[str, float]] = None,
        dependency_matrix: Optional[np.ndarray] = None
    ) -> MultiConstraintPosterior:
        """
        Integrate multiple constraints using Bayesian framework.

        Args:
            constraint_data: Dict of {constraint_name: ecs_samples}
            constraint_weights: Optional weights for each constraint
            dependency_matrix: Optional correlation matrix between constraints

        Returns:
            MultiConstraintPosterior with combined results
        """
        constraint_names = list(constraint_data.keys())
        n_constraints = len(constraint_names)

        # Default: equal weights
        if constraint_weights is None:
            constraint_weights = {
                name: 1.0 / n_constraints for name in constraint_names
            }

        # Normalize weights
        total_weight = sum(constraint_weights.values())
        constraint_weights = {
            k: v / total_weight for k, v in constraint_weights.items()
        }

        # Get prior samples
        prior_samples = self._sample_prior(self.n_samples)

        # Calculate posterior using importance sampling
        # For each prior sample, calculate likelihood from each constraint
        log_likelihoods = np.zeros((self.n_samples, n_constraints))

        for i, name in enumerate(constraint_names):
            constraint_samples = constraint_data[name]
            weight = constraint_weights[name]

            # Estimate likelihood using kernel density estimation
            log_likelihoods[:, i] = self._log_likelihood_kde(
                prior_samples, constraint_samples, weight
            )

        # Account for dependencies if provided
        if dependency_matrix is not None:
            # Adjust likelihoods for correlations
            # This is a simplified approach; full treatment requires copulas
            log_likelihoods = self._adjust_for_dependencies(
                log_likelihoods, dependency_matrix
            )

        # Total log likelihood
        total_log_likelihood = np.sum(log_likelihoods, axis=1)

        # Importance weights
        log_weights = total_log_likelihood - np.max(total_log_likelihood)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Resample from posterior
        indices = np.random.choice(
            self.n_samples, size=self.n_samples, replace=True, p=weights
        )
        posterior_samples = prior_samples[indices]

        # Calculate statistics
        percentiles = {
            "5th": np.percentile(posterior_samples, 5),
            "16th": np.percentile(posterior_samples, 16),
            "50th": np.percentile(posterior_samples, 50),
            "84th": np.percentile(posterior_samples, 84),
            "95th": np.percentile(posterior_samples, 95)
        }

        # Calculate information gain (KL divergence from prior to posterior)
        information_gain = self._calculate_information_gain(
            prior_samples, posterior_samples
        )

        # Model evidence (marginal likelihood)
        evidence = np.log(np.mean(np.exp(log_weights)))

        return MultiConstraintPosterior(
            ecs_samples=posterior_samples,
            ecs_mean=np.mean(posterior_samples),
            ecs_median=percentiles["50th"],
            ecs_std=np.std(posterior_samples),
            ecs_percentiles=percentiles,
            constraint_weights=constraint_weights,
            information_gain=information_gain,
            evidence=evidence,
            metadata={
                "prior_type": self.prior_type,
                "n_constraints": n_constraints,
                "constraint_names": constraint_names,
                "n_samples": len(posterior_samples)
            }
        )

    def _sample_prior(self, n_samples: int) -> np.ndarray:
        """
        Sample from prior distribution.

        Args:
            n_samples: Number of samples

        Returns:
            Array of ECS samples from prior
        """
        if self.prior_type == "uniform":
            # Uniform prior over reasonable ECS range
            low = self.prior_params.get("low", 0.5)
            high = self.prior_params.get("high", 10.0)
            return np.random.uniform(low, high, n_samples)

        elif self.prior_type == "jeffreys":
            # Jeffreys prior: p(ECS) ∝ 1/ECS (scale-invariant)
            # Sample from log-uniform distribution
            log_low = np.log(self.prior_params.get("low", 0.5))
            log_high = np.log(self.prior_params.get("high", 10.0))
            log_samples = np.random.uniform(log_low, log_high, n_samples)
            return np.exp(log_samples)

        elif self.prior_type == "informed":
            # Weakly informed prior based on CMIP ensemble
            mean = self.prior_params.get("mean", 3.0)
            std = self.prior_params.get("std", 1.5)
            samples = np.random.normal(mean, std, n_samples)
            # Truncate at physical bounds
            return np.clip(samples, 0.5, 10.0)

        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def _log_likelihood_kde(
        self,
        test_points: np.ndarray,
        constraint_samples: np.ndarray,
        weight: float = 1.0,
        bandwidth: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate log likelihood using kernel density estimation.

        Args:
            test_points: Points at which to evaluate likelihood
            constraint_samples: Samples from constraint distribution
            weight: Weight for this constraint
            bandwidth: KDE bandwidth (default: Scott's rule)

        Returns:
            Log likelihood values at test points
        """
        n = len(constraint_samples)

        if bandwidth is None:
            # Scott's rule
            bandwidth = n**(-1.0/5.0) * np.std(constraint_samples)

        # Gaussian kernel density estimation
        # log p(x) = log(1/n ∑ K((x - xᵢ)/h))
        diff = test_points[:, np.newaxis] - constraint_samples[np.newaxis, :]
        kernel_values = -0.5 * (diff / bandwidth)**2 - 0.5 * np.log(2 * np.pi) - np.log(bandwidth)

        # Log-sum-exp trick for numerical stability
        max_vals = np.max(kernel_values, axis=1)
        log_likelihood = max_vals + np.log(
            np.mean(np.exp(kernel_values - max_vals[:, np.newaxis]), axis=1)
        )

        # Apply weight
        return weight * log_likelihood

    def _adjust_for_dependencies(
        self,
        log_likelihoods: np.ndarray,
        dependency_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Adjust log likelihoods for dependencies between constraints.

        This is a simplified approach. A full treatment would require
        modeling the joint distribution using copulas or similar.

        Args:
            log_likelihoods: Log likelihoods for each constraint (n_samples, n_constraints)
            dependency_matrix: Correlation matrix between constraints

        Returns:
            Adjusted log likelihoods
        """
        # Downweight highly correlated constraints
        # Independent constraints: full weight
        # Perfectly correlated: treat as single constraint

        n_constraints = log_likelihoods.shape[1]
        adjusted = log_likelihoods.copy()

        # Calculate effective weights based on correlation structure
        # Use eigenvalues of correlation matrix as measure of independence
        eigenvalues = np.linalg.eigvalsh(dependency_matrix)
        effective_n = np.sum(eigenvalues) / np.max(eigenvalues)

        # Scale each constraint's contribution
        scale_factor = effective_n / n_constraints
        adjusted *= scale_factor

        return adjusted

    def _calculate_information_gain(
        self,
        prior_samples: np.ndarray,
        posterior_samples: np.ndarray
    ) -> float:
        """
        Calculate information gain from prior to posterior (KL divergence).

        Args:
            prior_samples: Samples from prior
            posterior_samples: Samples from posterior

        Returns:
            Information gain in bits
        """
        # Use histogram-based approximation
        bins = np.linspace(0, 10, 50)

        prior_hist, _ = np.histogram(prior_samples, bins=bins, density=True)
        posterior_hist, _ = np.histogram(posterior_samples, bins=bins, density=True)

        # Add small value to avoid log(0)
        epsilon = 1e-10
        prior_hist += epsilon
        posterior_hist += epsilon

        # Normalize
        prior_hist /= np.sum(prior_hist)
        posterior_hist /= np.sum(posterior_hist)

        # KL divergence: D_KL(P||Q) = ∑ P(x) log(P(x)/Q(x))
        kl_div = np.sum(posterior_hist * np.log2(posterior_hist / prior_hist))

        return max(0, kl_div)  # Ensure non-negative


def calculate_bayes_factors(
    posteriors: List[MultiConstraintPosterior],
    model_names: List[str]
) -> Dict[str, float]:
    """
    Calculate Bayes factors for model comparison.

    Args:
        posteriors: List of posterior results from different models
        model_names: Names of models for comparison

    Returns:
        Dict of Bayes factors relative to first model
    """
    reference_evidence = posteriors[0].evidence
    bayes_factors = {}

    for i, (name, posterior) in enumerate(zip(model_names, posteriors)):
        if i == 0:
            bayes_factors[name] = 1.0  # Reference
        else:
            # BF = p(D|M₁) / p(D|M₀)
            bayes_factors[name] = np.exp(posterior.evidence - reference_evidence)

    return bayes_factors
