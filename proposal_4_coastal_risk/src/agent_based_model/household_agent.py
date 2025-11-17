"""
Household Agent

Individual household agent with decision-making capabilities for
risk perception, adaptation, and evacuation.
"""

import numpy as np
from mesa import Agent
from typing import Optional, Tuple


class HouseholdAgent(Agent):
    """
    Represents a household in the coastal vulnerability model.

    Each household has characteristics (wealth, size, education),
    makes decisions about adaptation, and experiences hazard impacts.

    Parameters
    ----------
    unique_id : int
        Unique identifier for the agent
    model : Model
        Reference to the parent model
    wealth : float
        Household wealth (USD)
    household_size : int
        Number of people in household
    education_level : str
        Education level: 'low', 'medium', or 'high'
    """

    def __init__(
        self,
        unique_id: int,
        model,
        wealth: float,
        household_size: int,
        education_level: str
    ):
        super().__init__(unique_id, model)

        # Household characteristics
        self.wealth = wealth
        self.household_size = household_size
        self.education_level = education_level

        # State variables
        self.risk_perception = 0.5  # 0-1 scale
        self.has_adapted = False
        self.is_displaced = False
        self.damage = 0.0

        # Adaptation options
        self.adaptation_cost = self._calculate_adaptation_cost()
        self.years_of_experience = 0
        self.previous_flood_exposure = 0

        # Decision parameters
        self.risk_aversion = np.random.beta(2, 2)  # 0-1, mean 0.5
        self.discount_rate = 0.05  # Annual discount rate

    def _calculate_adaptation_cost(self) -> float:
        """
        Calculate cost of adaptation measures.

        Includes elevation, flood-proofing, insurance, etc.
        """
        # Base cost varies by wealth (ability to invest)
        base_cost = self.wealth * 0.15  # 15% of wealth

        # Economies of scale for wealthier households
        if self.wealth > 10000:
            base_cost *= 0.8

        return max(base_cost, 500)  # Minimum cost

    def step(self):
        """
        Execute one step of agent behavior.

        1. Update risk perception
        2. Make adaptation decision
        3. Calculate flood damage if exposed
        4. Update state
        """
        # Update risk perception based on environment
        self._update_risk_perception()

        # Make adaptation decision
        if not self.has_adapted:
            self._decide_adaptation()

        # Calculate damage from current flooding
        self._calculate_damage()

        # Update experience
        self.years_of_experience += 1 / 365  # Daily steps

    def _update_risk_perception(self):
        """
        Update risk perception based on:
        - Flood exposure at current location
        - Previous experience
        - Education level
        - Social network (neighboring agents)
        """
        # Get flood depth at current location
        x, y = self.pos
        flood_depth = 0

        if self.model.flood_depth is not None:
            flood_depth = self.model.flood_depth[y, x]

        # Direct experience component
        if flood_depth > 0:
            self.previous_flood_exposure += 1
            experience_factor = min(self.previous_flood_exposure * 0.1, 0.5)
        else:
            experience_factor = self.previous_flood_exposure * 0.05

        # Education component
        education_factors = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7
        }
        education_factor = education_factors[self.education_level]

        # Social learning from neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=2
        )

        if len(neighbors) > 0:
            neighbor_perception = np.mean([n.risk_perception for n in neighbors])
            social_factor = neighbor_perception * 0.2
        else:
            social_factor = 0

        # Physical exposure component
        if self.model.elevation_map is not None:
            elevation = self.model.elevation_map[y, x]
            exposure_factor = max(0, (5 - elevation) / 5)  # Higher risk at low elevation
        else:
            exposure_factor = 0.5

        # Combine factors with weighted average
        self.risk_perception = (
            0.3 * experience_factor +
            0.2 * education_factor +
            0.2 * social_factor +
            0.3 * exposure_factor
        )

        # Bound between 0 and 1
        self.risk_perception = np.clip(self.risk_perception, 0, 1)

    def _decide_adaptation(self):
        """
        Decide whether to implement adaptation measures.

        Based on:
        - Risk perception
        - Wealth (ability to pay)
        - Risk aversion
        - Expected benefits vs costs
        """
        # Can afford adaptation?
        can_afford = self.wealth >= self.adaptation_cost

        if not can_afford:
            return

        # Expected benefit (avoided damage)
        expected_annual_damage = self._estimate_expected_damage()
        years_to_consider = 10  # Planning horizon

        # Net present value of adaptation
        npv = 0
        for t in range(years_to_consider):
            discount_factor = (1 + self.discount_rate) ** (-t)
            npv += expected_annual_damage * discount_factor

        npv -= self.adaptation_cost

        # Decision rule: adapt if NPV > 0 and risk perception is high
        threshold = 0.5 - (self.risk_aversion * 0.3)

        if npv > 0 and self.risk_perception > threshold:
            self.has_adapted = True
            self.wealth -= self.adaptation_cost

    def _estimate_expected_damage(self) -> float:
        """
        Estimate expected annual damage without adaptation.

        Uses simplified damage function based on flood depth.
        """
        x, y = self.pos

        if self.model.elevation_map is None:
            return 0

        elevation = self.model.elevation_map[y, x]

        # Probability of flooding at different intensities
        # (simplified - use full probability distribution in practice)
        prob_minor = 0.1  # 10% annual chance
        prob_major = 0.02  # 2% annual chance

        # Damage as fraction of wealth
        if elevation < 2:  # Very exposed
            damage_minor = self.wealth * 0.1
            damage_major = self.wealth * 0.5
        elif elevation < 5:  # Moderately exposed
            damage_minor = self.wealth * 0.05
            damage_major = self.wealth * 0.2
        else:  # Less exposed
            damage_minor = self.wealth * 0.02
            damage_major = self.wealth * 0.1

        expected_damage = (
            prob_minor * damage_minor +
            prob_major * damage_major
        )

        return expected_damage

    def _calculate_damage(self):
        """
        Calculate actual damage from current flood event.

        Damage depends on:
        - Flood depth
        - Household wealth (proxy for asset value)
        - Adaptation status
        """
        x, y = self.pos

        if self.model.flood_depth is None:
            self.damage = 0
            return

        flood_depth = self.model.flood_depth[y, x]

        if flood_depth <= 0:
            self.damage = 0
            return

        # Depth-damage function
        if flood_depth < 0.5:
            damage_fraction = 0.1 * flood_depth
        elif flood_depth < 1.0:
            damage_fraction = 0.2 + 0.3 * (flood_depth - 0.5)
        elif flood_depth < 2.0:
            damage_fraction = 0.35 + 0.25 * (flood_depth - 1.0)
        else:
            damage_fraction = 0.6 + 0.2 * min(flood_depth - 2.0, 2.0)

        # Cap at total wealth
        damage_fraction = min(damage_fraction, 1.0)

        # Adaptation reduces damage
        if self.has_adapted:
            damage_fraction *= 0.3  # 70% reduction

        self.damage = self.wealth * damage_fraction

        # Displacement threshold
        if self.damage > self.wealth * 0.5:
            self.is_displaced = True

        # Reduce wealth by damage
        self.wealth -= self.damage

    def get_vulnerability_score(self) -> float:
        """
        Calculate overall vulnerability score.

        Combines exposure, sensitivity, and adaptive capacity.

        Returns
        -------
        float
            Vulnerability score (0-1, higher = more vulnerable)
        """
        # Exposure component
        x, y = self.pos
        if self.model.elevation_map is not None:
            elevation = self.model.elevation_map[y, x]
            exposure = max(0, (5 - elevation) / 5)
        else:
            exposure = 0.5

        # Sensitivity component (household characteristics)
        wealth_sensitivity = 1 - min(self.wealth / 10000, 1)  # Poorer = more sensitive
        size_sensitivity = min(self.household_size / 6, 1)  # Larger = more sensitive

        sensitivity = (wealth_sensitivity + size_sensitivity) / 2

        # Adaptive capacity component
        if self.has_adapted:
            adaptive_capacity = 0.8
        else:
            # Based on wealth and education
            wealth_capacity = min(self.wealth / 5000, 1)
            education_capacity = {'low': 0.3, 'medium': 0.6, 'high': 0.9}[self.education_level]
            adaptive_capacity = (wealth_capacity + education_capacity) / 2

        # Vulnerability = (Exposure × Sensitivity) / Adaptive Capacity
        # Normalized to 0-1
        vulnerability = (exposure * sensitivity) / max(adaptive_capacity, 0.1)
        vulnerability = min(vulnerability, 1.0)

        return vulnerability

    def reset(self):
        """Reset agent state for new scenario."""
        self.is_displaced = False
        self.damage = 0.0
        # Note: Keep risk_perception and has_adapted for learning across scenarios
