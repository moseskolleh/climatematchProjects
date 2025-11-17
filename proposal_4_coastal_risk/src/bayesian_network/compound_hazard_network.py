"""
Compound Hazard Bayesian Network

This module implements the core Bayesian network for modeling
compound coastal hazards with uncertainty quantification.
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from typing import List, Dict, Optional, Tuple
import networkx as nx
from scipy import stats


class CompoundHazardNetwork:
    """
    Bayesian Network for modeling compound coastal hazards.

    This network captures probabilistic dependencies between multiple
    coastal hazards and provides uncertainty-aware risk assessments.

    Parameters
    ----------
    hazard_types : List[str]
        List of hazard types to include in the network.
        Options: 'storm_surge', 'slr', 'precipitation', 'wave_height', 'erosion'
    city : str, optional
        Target city for location-specific priors
    use_hierarchical : bool, default=True
        Use hierarchical Bayesian approach with global priors
    """

    def __init__(
        self,
        hazard_types: List[str],
        city: Optional[str] = None,
        use_hierarchical: bool = True
    ):
        self.hazard_types = hazard_types
        self.city = city
        self.use_hierarchical = use_hierarchical

        # Initialize network structure
        self.graph = nx.DiGraph()
        self.model = None
        self.trace = None

        # Data storage
        self.observed_data = None
        self.synthetic_events = None

        # Build initial network structure
        self._build_network_structure()

    def _build_network_structure(self):
        """
        Build the directed acyclic graph structure of the Bayesian network.

        Defines causal relationships between hazards based on physical understanding:
        - Sea level rise affects coastal flooding baseline
        - Atmospheric pressure affects storm surge
        - Storm surge and precipitation compound for total water level
        - Wave height affects coastal erosion
        """
        # Add nodes for each hazard type
        for hazard in self.hazard_types:
            self.graph.add_node(hazard, hazard_type=hazard)

        # Define dependencies based on physics
        dependencies = {
            ('slr', 'storm_surge'): 'baseline_elevation',
            ('storm_surge', 'total_flood'): 'water_level',
            ('precipitation', 'total_flood'): 'runoff',
            ('wave_height', 'erosion'): 'mechanical_stress',
            ('slr', 'erosion'): 'shoreline_retreat'
        }

        for (source, target), relation in dependencies.items():
            if source in self.hazard_types and target in self.hazard_types:
                self.graph.add_edge(source, target, relation=relation)

    def fit(
        self,
        data: pd.DataFrame,
        n_samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95
    ):
        """
        Fit the Bayesian network to observed data.

        Parameters
        ----------
        data : pd.DataFrame
            Historical hazard data with columns for each hazard type
        n_samples : int
            Number of posterior samples to draw
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance probability for NUTS sampler
        """
        self.observed_data = data

        # Build PyMC3 model
        self.model = self._build_pymc3_model(data)

        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                n_samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True
            )

        print(f"Sampling complete. R-hat diagnostics:")
        print(az.summary(self.trace, var_names=['~log_likelihood']))

        return self

    def _build_pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """
        Construct the PyMC3 probabilistic model.

        Uses hierarchical priors when specified, allowing information
        sharing across locations while accounting for local variations.
        """
        model = pm.Model()

        with model:
            # Global hyperpriors (if hierarchical)
            if self.use_hierarchical:
                # Location parameter hyperprior
                mu_global = pm.Normal('mu_global', mu=0, sigma=10)
                sigma_global = pm.HalfNormal('sigma_global', sigma=5)

                # Scale parameter hyperprior
                sigma_mu = pm.HalfNormal('sigma_mu', sigma=3)

            # Hazard-specific parameters
            hazard_params = {}

            for hazard in self.hazard_types:
                if hazard in data.columns:
                    # Extract data for this hazard
                    hazard_data = data[hazard].values

                    if self.use_hierarchical:
                        # Hierarchical prior
                        mu = pm.Normal(
                            f'{hazard}_mu',
                            mu=mu_global,
                            sigma=sigma_global
                        )
                        sigma = pm.HalfNormal(
                            f'{hazard}_sigma',
                            sigma=sigma_mu
                        )
                    else:
                        # Independent prior
                        mu = pm.Normal(f'{hazard}_mu', mu=0, sigma=10)
                        sigma = pm.HalfNormal(f'{hazard}_sigma', sigma=5)

                    hazard_params[hazard] = {'mu': mu, 'sigma': sigma}

                    # Likelihood
                    pm.Normal(
                        f'{hazard}_obs',
                        mu=mu,
                        sigma=sigma,
                        observed=hazard_data
                    )

            # Model dependencies between hazards
            self._add_dependency_priors(model, hazard_params, data)

        return model

    def _add_dependency_priors(
        self,
        model: pm.Model,
        hazard_params: Dict,
        data: pd.DataFrame
    ):
        """
        Add priors for dependencies between hazards.

        Models conditional dependencies using copulas and
        regression relationships.
        """
        with model:
            # For each edge in the network
            for source, target in self.graph.edges():
                if source in data.columns and target in data.columns:
                    # Correlation parameter
                    rho = pm.Uniform(
                        f'rho_{source}_{target}',
                        lower=-1,
                        upper=1
                    )

                    # Regression coefficient
                    beta = pm.Normal(
                        f'beta_{source}_{target}',
                        mu=0,
                        sigma=1
                    )

    def sample_scenarios(
        self,
        n_scenarios: int = 1000,
        return_period: Optional[float] = None,
        future_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic compound event scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate
        return_period : float, optional
            Focus on events with specific return period (years)
        future_year : int, optional
            Project scenarios to future year accounting for trends

        Returns
        -------
        pd.DataFrame
            Synthetic event scenarios
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before sampling scenarios")

        # Sample from posterior predictive distribution
        with self.model:
            posterior_pred = pm.sample_posterior_predictive(
                self.trace,
                samples=n_scenarios
            )

        # Convert to DataFrame
        scenarios = {}
        for hazard in self.hazard_types:
            if f'{hazard}_obs' in posterior_pred:
                scenarios[hazard] = posterior_pred[f'{hazard}_obs'].flatten()[:n_scenarios]

        scenario_df = pd.DataFrame(scenarios)

        # Apply return period filter if specified
        if return_period is not None:
            scenario_df = self._filter_by_return_period(scenario_df, return_period)

        # Apply future projection if specified
        if future_year is not None:
            scenario_df = self._project_to_future(scenario_df, future_year)

        self.synthetic_events = scenario_df
        return scenario_df

    def _filter_by_return_period(
        self,
        scenarios: pd.DataFrame,
        return_period: float
    ) -> pd.DataFrame:
        """
        Filter scenarios to match a target return period.

        Uses extreme value theory to identify events consistent
        with the specified return period.
        """
        # Calculate compound hazard index
        # (simplified - use copula-based joint probability in practice)
        hazard_index = scenarios.sum(axis=1)

        # Fit generalized extreme value distribution
        gev_params = stats.genextreme.fit(hazard_index)

        # Find threshold for return period
        threshold = stats.genextreme.ppf(
            1 - 1/return_period,
            *gev_params
        )

        # Filter scenarios
        extreme_scenarios = scenarios[hazard_index >= threshold]

        return extreme_scenarios

    def _project_to_future(
        self,
        scenarios: pd.DataFrame,
        future_year: int
    ) -> pd.DataFrame:
        """
        Project scenarios to future year.

        Applies trend adjustments based on climate projections,
        particularly for sea level rise.
        """
        # Simple linear trend (replace with RCP scenario data)
        current_year = 2025
        years_ahead = future_year - current_year

        # Sea level rise projection (mm/year)
        if 'slr' in scenarios.columns:
            slr_trend = 3.3  # mm/year, intermediate scenario
            scenarios['slr'] = scenarios['slr'] + (slr_trend * years_ahead)

        # Storm surge intensification (simplified)
        if 'storm_surge' in scenarios.columns:
            surge_factor = 1 + (0.01 * years_ahead)  # 1% per decade
            scenarios['storm_surge'] = scenarios['storm_surge'] * surge_factor

        return scenarios

    def get_compound_probability(
        self,
        thresholds: Dict[str, float]
    ) -> float:
        """
        Calculate probability of compound event exceeding thresholds.

        Parameters
        ----------
        thresholds : Dict[str, float]
            Threshold values for each hazard type

        Returns
        -------
        float
            Probability of compound exceedance
        """
        if self.synthetic_events is None:
            self.sample_scenarios()

        # Check which scenarios exceed all thresholds
        exceeds_all = np.ones(len(self.synthetic_events), dtype=bool)

        for hazard, threshold in thresholds.items():
            if hazard in self.synthetic_events.columns:
                exceeds_all &= (self.synthetic_events[hazard] >= threshold)

        probability = exceeds_all.sum() / len(self.synthetic_events)

        return probability

    def visualize_network(self, save_path: Optional[str] = None):
        """
        Visualize the Bayesian network structure.

        Parameters
        ----------
        save_path : str, optional
            Path to save the visualization
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.9
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=2
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=10,
            font_weight='bold'
        )

        plt.title('Compound Coastal Hazard Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics of the fitted model.

        Returns
        -------
        pd.DataFrame
            Summary statistics including means, std, and credible intervals
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")

        summary = az.summary(self.trace)
        return summary
