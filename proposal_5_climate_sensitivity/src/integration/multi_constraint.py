"""
Multi-Constraint Framework

Bayesian integration of multiple independent constraints on climate sensitivity.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from scipy import stats
import warnings


class MultiConstraintFramework:
    """
    Bayesian framework for combining multiple climate sensitivity constraints

    Parameters:
    -----------
    constraints : list
        List of constraint objects or dictionaries with 'likelihood' and 'name'
    prior : str or callable, optional
        Prior distribution on ECS ('uniform', 'informed', or custom function)
    ecs_range : tuple, optional
        Range of ECS values to consider (default: (1.0, 6.0))
    test_independence : bool, optional
        Whether to test constraint independence (default: True)
    """

    def __init__(
        self,
        constraints: Optional[List] = None,
        prior: str = 'uniform',
        ecs_range: Tuple[float, float] = (1.0, 6.0),
        test_independence: bool = True
    ):
        self.constraints = constraints or []
        self.prior_type = prior
        self.ecs_range = ecs_range
        self.test_independence = test_independence

        # ECS values for integration
        self.ecs_values = np.linspace(ecs_range[0], ecs_range[1], 1000)

        # Results
        self.posterior = None
        self.independence_results = None

    def add_constraint(self, constraint) -> None:
        """
        Add a constraint to the framework

        Parameters:
        -----------
        constraint : object or dict
            Constraint with 'calculate_constraint' method or dict with 'likelihood'
        """
        self.constraints.append(constraint)

    def get_prior(self) -> np.ndarray:
        """
        Get prior distribution on ECS

        Returns:
        --------
        prior_pdf : np.ndarray
            Prior probability density
        """
        if self.prior_type == 'uniform':
            # Uniform prior over range
            prior_pdf = np.ones_like(self.ecs_values)

        elif self.prior_type == 'informed':
            # Weakly informative prior based on expert knowledge
            # Normal distribution centered at 3.0 K with large std dev
            prior_pdf = stats.norm.pdf(self.ecs_values, loc=3.0, scale=1.5)

        elif self.prior_type == 'jeffreys':
            # Jeffreys prior (1/ECS) - scale-invariant
            prior_pdf = 1.0 / self.ecs_values

        elif callable(self.prior_type):
            # Custom prior function
            prior_pdf = self.prior_type(self.ecs_values)

        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

        # Normalize
        prior_pdf /= np.trapz(prior_pdf, self.ecs_values)

        return prior_pdf

    def test_constraint_independence(self) -> Dict:
        """
        Test independence between constraints

        Returns:
        --------
        results : dict
            Independence test results for each pair
        """
        from .independence_test import IndependenceTest

        if len(self.constraints) < 2:
            return {}

        tester = IndependenceTest()
        results = {}

        for i, c1 in enumerate(self.constraints):
            for j, c2 in enumerate(self.constraints[i+1:], start=i+1):
                constraint1 = c1.calculate_constraint() if hasattr(c1, 'calculate_constraint') else c1
                constraint2 = c2.calculate_constraint() if hasattr(c2, 'calculate_constraint') else c2

                name1 = constraint1.get('name', f'constraint_{i}')
                name2 = constraint2.get('name', f'constraint_{j}')

                # Test independence using samples
                independence = tester.test_sample_independence(
                    constraint1.get('ecs_samples'),
                    constraint2.get('ecs_samples')
                )

                results[f"{name1}_vs_{name2}"] = independence

        return results

    def integrate_constraints(
        self,
        method: str = 'bayesian',
        weights: Optional[np.ndarray] = None
    ) -> 'ECSSDistribution':
        """
        Combine constraints to produce posterior ECS distribution

        Parameters:
        -----------
        method : str
            Integration method ('bayesian', 'equal_weight', 'weighted')
        weights : np.ndarray, optional
            Custom weights for each constraint (used if method='weighted')

        Returns:
        --------
        posterior_dist : ECSDistribution
            Posterior ECS distribution object
        """
        # Get prior
        prior_pdf = self.get_prior()

        if method == 'bayesian':
            # Full Bayesian integration
            posterior_pdf = prior_pdf.copy()

            for constraint in self.constraints:
                # Get constraint data
                if hasattr(constraint, 'calculate_constraint'):
                    constraint_data = constraint.calculate_constraint()
                else:
                    constraint_data = constraint

                # Get likelihood function
                likelihood_func = constraint_data.get('likelihood')

                if likelihood_func is None:
                    warnings.warn(f"No likelihood function for {constraint_data.get('name')}")
                    continue

                # Evaluate likelihood
                likelihood = likelihood_func(self.ecs_values)

                # Multiply posterior by likelihood
                posterior_pdf *= likelihood

            # Normalize
            posterior_pdf /= np.trapz(posterior_pdf, self.ecs_values)

        elif method == 'equal_weight':
            # Equally weight all constraint likelihoods
            combined_likelihood = np.ones_like(self.ecs_values)

            for constraint in self.constraints:
                constraint_data = constraint.calculate_constraint() if hasattr(constraint, 'calculate_constraint') else constraint
                likelihood_func = constraint_data.get('likelihood')

                if likelihood_func:
                    combined_likelihood *= likelihood_func(self.ecs_values)

            posterior_pdf = prior_pdf * combined_likelihood
            posterior_pdf /= np.trapz(posterior_pdf, self.ecs_values)

        elif method == 'weighted':
            # Weight constraints by provided weights
            if weights is None:
                raise ValueError("Weights must be provided for weighted method")

            if len(weights) != len(self.constraints):
                raise ValueError("Number of weights must match number of constraints")

            combined_likelihood = np.ones_like(self.ecs_values)

            for i, constraint in enumerate(self.constraints):
                constraint_data = constraint.calculate_constraint() if hasattr(constraint, 'calculate_constraint') else constraint
                likelihood_func = constraint_data.get('likelihood')

                if likelihood_func:
                    likelihood = likelihood_func(self.ecs_values)
                    # Weight the likelihood
                    weighted_likelihood = likelihood ** weights[i]
                    combined_likelihood *= weighted_likelihood

            posterior_pdf = prior_pdf * combined_likelihood
            posterior_pdf /= np.trapz(posterior_pdf, self.ecs_values)

        else:
            raise ValueError(f"Unknown integration method: {method}")

        # Store posterior
        self.posterior = posterior_pdf

        # Test independence if requested
        if self.test_independence and len(self.constraints) > 1:
            self.independence_results = self.test_constraint_independence()

        # Return distribution object
        return ECSDistribution(self.ecs_values, posterior_pdf, self.constraints)

    def sensitivity_analysis(self) -> Dict:
        """
        Perform sensitivity analysis to methodological choices

        Returns:
        --------
        results : dict
            Sensitivity test results
        """
        results = {}

        # Test different priors
        for prior in ['uniform', 'informed', 'jeffreys']:
            original_prior = self.prior_type
            self.prior_type = prior

            posterior = self.integrate_constraints()
            results[f'prior_{prior}'] = {
                'median': posterior.median(),
                'ci_90': posterior.credible_interval(0.90)
            }

            self.prior_type = original_prior

        # Test leaving out each constraint (jackknife)
        for i, constraint in enumerate(self.constraints):
            # Temporarily remove constraint
            removed = self.constraints.pop(i)

            posterior = self.integrate_constraints()
            constraint_name = removed.calculate_constraint()['name'] if hasattr(removed, 'calculate_constraint') else removed.get('name', f'constraint_{i}')

            results[f'without_{constraint_name}'] = {
                'median': posterior.median(),
                'ci_90': posterior.credible_interval(0.90)
            }

            # Restore constraint
            self.constraints.insert(i, removed)

        return results

    def __repr__(self) -> str:
        return f"MultiConstraintFramework(n_constraints={len(self.constraints)}, prior='{self.prior_type}')"


class ECSDistribution:
    """
    ECS probability distribution with convenience methods

    Parameters:
    -----------
    ecs_values : np.ndarray
        ECS values
    pdf : np.ndarray
        Probability density function
    constraints : list
        Original constraints used
    """

    def __init__(self, ecs_values: np.ndarray, pdf: np.ndarray, constraints: List):
        self.ecs_values = ecs_values
        self.pdf = pdf
        self.constraints = constraints

        # Calculate CDF for percentile calculations
        self.cdf = np.cumsum(pdf * np.diff(ecs_values, prepend=ecs_values[0]))
        self.cdf /= self.cdf[-1]  # Normalize

    def median(self) -> float:
        """Return median ECS"""
        idx = np.argmin(np.abs(self.cdf - 0.5))
        return self.ecs_values[idx]

    def mean(self) -> float:
        """Return mean ECS"""
        return np.trapz(self.ecs_values * self.pdf, self.ecs_values)

    def mode(self) -> float:
        """Return mode (most likely) ECS"""
        idx = np.argmax(self.pdf)
        return self.ecs_values[idx]

    def percentile(self, percentiles: List[float]) -> np.ndarray:
        """
        Return specified percentiles

        Parameters:
        -----------
        percentiles : list
            List of percentiles (0-100)

        Returns:
        --------
        values : np.ndarray
            ECS values at percentiles
        """
        percentiles = np.array(percentiles) / 100.0
        indices = [np.argmin(np.abs(self.cdf - p)) for p in percentiles]
        return self.ecs_values[indices]

    def credible_interval(self, probability: float) -> Tuple[float, float]:
        """
        Return credible interval

        Parameters:
        -----------
        probability : float
            Probability level (e.g., 0.90 for 90% CI)

        Returns:
        --------
        lower, upper : float
            Bounds of credible interval
        """
        tail = (1 - probability) / 2
        lower_percentile = tail * 100
        upper_percentile = (1 - tail) * 100

        values = self.percentile([lower_percentile, upper_percentile])
        return tuple(values)

    def sample(self, n_samples: int = 10000) -> np.ndarray:
        """
        Draw samples from posterior distribution

        Parameters:
        -----------
        n_samples : int
            Number of samples to draw

        Returns:
        --------
        samples : np.ndarray
            ECS samples
        """
        # Inverse transform sampling
        uniform_samples = np.random.uniform(0, 1, n_samples)
        samples = np.interp(uniform_samples, self.cdf, self.ecs_values)
        return samples

    def plot(self, ax=None):
        """
        Plot posterior distribution

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot PDF
        ax.plot(self.ecs_values, self.pdf, 'k-', linewidth=2, label='Posterior')
        ax.fill_between(self.ecs_values, 0, self.pdf, alpha=0.3)

        # Mark median and CI
        median = self.median()
        ci_66 = self.credible_interval(0.66)
        ci_90 = self.credible_interval(0.90)

        ax.axvline(median, color='r', linestyle='--', label=f'Median: {median:.2f} K')
        ax.axvspan(ci_66[0], ci_66[1], alpha=0.2, color='orange', label=f'66% CI: [{ci_66[0]:.2f}, {ci_66[1]:.2f}] K')
        ax.axvspan(ci_90[0], ci_90[1], alpha=0.1, color='blue', label=f'90% CI: [{ci_90[0]:.2f}, {ci_90[1]:.2f}] K')

        ax.set_xlabel('Equilibrium Climate Sensitivity (K)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Posterior ECS Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self) -> str:
        median = self.median()
        ci_90 = self.credible_interval(0.90)
        return f"ECSDistribution(median={median:.2f} K, 90% CI=[{ci_90[0]:.2f}, {ci_90[1]:.2f}] K)"
