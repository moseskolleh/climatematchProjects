"""
Deep Uncertainty Analysis

Implements methods for decision-making under deep uncertainty,
focusing on robustness rather than precise probabilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple
from itertools import product
from scipy import stats


class DeepUncertaintyAnalysis:
    """
    Deep uncertainty analysis for coastal risk decisions.

    Uses robust decision making (RDM) framework to identify strategies
    that perform well across multiple plausible futures.

    Parameters
    ----------
    n_scenarios : int
        Number of scenarios to explore
    objectives : List[str]
        Decision objectives to optimize
    """

    def __init__(
        self,
        n_scenarios: int = 1000,
        objectives: Optional[List[str]] = None
    ):
        self.n_scenarios = n_scenarios

        if objectives is None:
            self.objectives = [
                'minimize_casualties',
                'minimize_economic_loss',
                'minimize_disruption'
            ]
        else:
            self.objectives = objectives

        self.scenarios = None
        self.strategies = []
        self.performance_matrix = None

    def generate_uncertainty_space(
        self,
        uncertain_factors: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Generate scenarios spanning uncertainty space.

        Uses Latin Hypercube Sampling for efficient coverage.

        Parameters
        ----------
        uncertain_factors : Dict[str, Tuple[float, float]]
            Dictionary of uncertain factors with (min, max) ranges

        Returns
        -------
        pd.DataFrame
            Scenario matrix
        """
        from scipy.stats import qmc

        n_factors = len(uncertain_factors)
        sampler = qmc.LatinHypercube(d=n_factors)
        samples = sampler.random(n=self.n_scenarios)

        # Scale to factor ranges
        scenarios_dict = {}
        for i, (factor, (min_val, max_val)) in enumerate(uncertain_factors.items()):
            scenarios_dict[factor] = min_val + samples[:, i] * (max_val - min_val)

        self.scenarios = pd.DataFrame(scenarios_dict)
        return self.scenarios

    def define_adaptation_strategy(
        self,
        name: str,
        actions: List[Dict],
        cost: float,
        implementation_time: int
    ):
        """
        Define an adaptation strategy.

        Parameters
        ----------
        name : str
            Strategy name
        actions : List[Dict]
            List of adaptation actions
        cost : float
            Implementation cost
        implementation_time : int
            Time to implement (years)
        """
        strategy = {
            'name': name,
            'actions': actions,
            'cost': cost,
            'implementation_time': implementation_time
        }

        self.strategies.append(strategy)

    def evaluate_strategies(
        self,
        scenarios: pd.DataFrame,
        impact_model: Callable
    ) -> pd.DataFrame:
        """
        Evaluate all strategies across all scenarios.

        Parameters
        ----------
        scenarios : pd.DataFrame
            Scenario matrix
        impact_model : Callable
            Function that calculates impacts for a scenario and strategy

        Returns
        -------
        pd.DataFrame
            Performance matrix (strategies × scenarios × objectives)
        """
        results = []

        for strategy in self.strategies:
            for idx, scenario in scenarios.iterrows():
                # Calculate performance for each objective
                performance = impact_model(scenario, strategy)

                results.append({
                    'strategy': strategy['name'],
                    'scenario_id': idx,
                    **performance
                })

        self.performance_matrix = pd.DataFrame(results)
        return self.performance_matrix

    def identify_robust_strategies(
        self,
        scenarios: pd.DataFrame,
        objectives: Optional[List[str]] = None,
        robustness_metric: str = 'regret'
    ) -> pd.DataFrame:
        """
        Identify robust strategies across scenarios.

        Parameters
        ----------
        scenarios : pd.DataFrame
            Hazard scenarios
        objectives : List[str], optional
            Objectives to consider
        robustness_metric : str
            'regret', 'percentile', or 'satisficing'

        Returns
        -------
        pd.DataFrame
            Strategy rankings with robustness scores
        """
        if objectives is None:
            objectives = self.objectives

        if self.performance_matrix is None:
            raise ValueError("Must evaluate strategies first")

        # Calculate robustness for each strategy
        robustness_scores = []

        for strategy_name in self.performance_matrix['strategy'].unique():
            strategy_data = self.performance_matrix[
                self.performance_matrix['strategy'] == strategy_name
            ]

            if robustness_metric == 'regret':
                score = self._calculate_regret(strategy_data, objectives)
            elif robustness_metric == 'percentile':
                score = self._calculate_percentile_performance(strategy_data, objectives)
            elif robustness_metric == 'satisficing':
                score = self._calculate_satisficing(strategy_data, objectives)
            else:
                raise ValueError(f"Unknown robustness metric: {robustness_metric}")

            robustness_scores.append({
                'strategy': strategy_name,
                'robustness_score': score
            })

        rankings = pd.DataFrame(robustness_scores).sort_values(
            'robustness_score',
            ascending=False
        )

        return rankings

    def _calculate_regret(
        self,
        strategy_data: pd.DataFrame,
        objectives: List[str]
    ) -> float:
        """
        Calculate minimax regret for strategy.

        Regret = difference from best possible outcome in each scenario
        """
        total_regret = 0

        for objective in objectives:
            if objective in strategy_data.columns:
                # For each scenario, calculate regret
                # (difference from best strategy in that scenario)
                scenario_regrets = []

                for scenario_id in strategy_data['scenario_id'].unique():
                    scenario_perf = self.performance_matrix[
                        self.performance_matrix['scenario_id'] == scenario_id
                    ]

                    # Best performance in this scenario
                    if 'minimize' in objective:
                        best_perf = scenario_perf[objective].min()
                        strategy_perf = strategy_data[
                            strategy_data['scenario_id'] == scenario_id
                        ][objective].values[0]
                        regret = strategy_perf - best_perf
                    else:  # maximize
                        best_perf = scenario_perf[objective].max()
                        strategy_perf = strategy_data[
                            strategy_data['scenario_id'] == scenario_id
                        ][objective].values[0]
                        regret = best_perf - strategy_perf

                    scenario_regrets.append(max(regret, 0))

                # Max regret across scenarios
                total_regret += max(scenario_regrets)

        # Lower regret is better, return negative for sorting
        return -total_regret

    def _calculate_percentile_performance(
        self,
        strategy_data: pd.DataFrame,
        objectives: List[str],
        percentile: float = 10
    ) -> float:
        """
        Calculate performance at worst-case percentile.

        Focuses on tail risk.
        """
        total_score = 0

        for objective in objectives:
            if objective in strategy_data.columns:
                if 'minimize' in objective:
                    # Worst case is high percentile
                    score = np.percentile(strategy_data[objective], 100 - percentile)
                    total_score -= score  # Negative because minimizing
                else:  # maximize
                    # Worst case is low percentile
                    score = np.percentile(strategy_data[objective], percentile)
                    total_score += score

        return total_score

    def _calculate_satisficing(
        self,
        strategy_data: pd.DataFrame,
        objectives: List[str],
        thresholds: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate fraction of scenarios meeting performance thresholds.
        """
        if thresholds is None:
            # Use median performance as threshold
            thresholds = {}
            for obj in objectives:
                if obj in strategy_data.columns:
                    thresholds[obj] = strategy_data[obj].median()

        # Check which scenarios meet all thresholds
        meets_thresholds = np.ones(len(strategy_data), dtype=bool)

        for objective, threshold in thresholds.items():
            if objective in strategy_data.columns:
                if 'minimize' in objective:
                    meets_thresholds &= (strategy_data[objective] <= threshold)
                else:  # maximize
                    meets_thresholds &= (strategy_data[objective] >= threshold)

        return meets_thresholds.sum() / len(strategy_data)

    def scenario_discovery(
        self,
        strategy: str,
        poor_performance_threshold: float,
        objective: str
    ) -> Dict:
        """
        Identify scenarios where strategy performs poorly.

        Uses Patient Rule Induction Method (PRIM) to find regions
        of uncertainty space associated with poor performance.

        Parameters
        ----------
        strategy : str
            Strategy name to analyze
        poor_performance_threshold : float
            Threshold defining poor performance
        objective : str
            Objective to consider

        Returns
        -------
        Dict
            Vulnerable conditions (ranges of uncertain factors)
        """
        if self.performance_matrix is None or self.scenarios is None:
            raise ValueError("Must evaluate strategies first")

        strategy_data = self.performance_matrix[
            self.performance_matrix['strategy'] == strategy
        ]

        # Identify poor performance scenarios
        if 'minimize' in objective:
            poor_scenarios = strategy_data[objective] > poor_performance_threshold
        else:
            poor_scenarios = strategy_data[objective] < poor_performance_threshold

        # Find characteristic ranges
        vulnerable_conditions = {}

        for factor in self.scenarios.columns:
            scenario_values = self.scenarios.loc[
                strategy_data['scenario_id'],
                factor
            ]

            poor_values = scenario_values[poor_scenarios]

            if len(poor_values) > 0:
                vulnerable_conditions[factor] = {
                    'range': (poor_values.min(), poor_values.max()),
                    'median': poor_values.median(),
                    'fraction_vulnerable': poor_scenarios.sum() / len(poor_scenarios)
                }

        return vulnerable_conditions

    def stress_test_strategy(
        self,
        strategy: str,
        extreme_scenarios: pd.DataFrame,
        impact_model: Callable
    ) -> pd.DataFrame:
        """
        Stress test a strategy under extreme scenarios.

        Parameters
        ----------
        strategy : str
            Strategy name
        extreme_scenarios : pd.DataFrame
            Extreme scenarios to test
        impact_model : Callable
            Impact calculation function

        Returns
        -------
        pd.DataFrame
            Stress test results
        """
        strategy_obj = next(s for s in self.strategies if s['name'] == strategy)

        results = []
        for idx, scenario in extreme_scenarios.iterrows():
            performance = impact_model(scenario, strategy_obj)
            results.append({
                'scenario_id': idx,
                **performance
            })

        return pd.DataFrame(results)

    def visualize_tradeoffs(
        self,
        objective1: str,
        objective2: str,
        save_path: Optional[str] = None
    ):
        """
        Visualize trade-offs between objectives.

        Creates scatter plot showing Pareto frontier.
        """
        import matplotlib.pyplot as plt

        if self.performance_matrix is None:
            raise ValueError("Must evaluate strategies first")

        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in self.performance_matrix['strategy'].unique():
            strategy_data = self.performance_matrix[
                self.performance_matrix['strategy'] == strategy
            ]

            ax.scatter(
                strategy_data[objective1],
                strategy_data[objective2],
                alpha=0.3,
                label=strategy
            )

        ax.set_xlabel(objective1.replace('_', ' ').title())
        ax.set_ylabel(objective2.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.title('Strategy Performance Trade-offs')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
