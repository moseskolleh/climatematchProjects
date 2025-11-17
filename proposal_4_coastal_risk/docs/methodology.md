# Methodology

## Overview

The Integrated Coastal Risk Framework combines three main components:

1. **Bayesian Network** for compound hazard modeling
2. **Agent-Based Model** for vulnerability assessment
3. **Deep Uncertainty Analysis** for decision support

## 1. Bayesian Network for Compound Hazards

### Conceptual Framework

The Bayesian network captures probabilistic dependencies between multiple coastal hazards:

```
Sea Level Rise ──┐
                 ├──> Total Water Level ──> Coastal Flooding
Storm Surge ─────┘

Precipitation ──────────────────────────> Pluvial Flooding
                                                 │
Wave Height ────────────────────────────────> Erosion
```

### Mathematical Formulation

For hazard variables H = {h₁, h₂, ..., hₙ}, the joint probability is:

P(H) = ∏ P(hᵢ | Parents(hᵢ))

Where Parents(hᵢ) are the direct causal predecessors in the network.

### Hierarchical Bayesian Approach

To handle data scarcity, we use hierarchical priors:

**Global Level:**
- μ_global ~ Normal(0, 10)
- σ_global ~ HalfNormal(5)

**Local Level:**
- μ_city ~ Normal(μ_global, σ_global)
- σ_city ~ HalfNormal(σ_mu)

**Observations:**
- hazard_obs ~ Normal(μ_city, σ_city)

This allows information sharing across cities while accommodating local variations.

### Dependency Modeling

Dependencies between hazards are modeled using:

1. **Correlation Structure**: Gaussian copulas for marginal correlation
2. **Regression Relationships**: For direct causal links
3. **Conditional Probabilities**: For discrete states

For example, storm surge conditional on atmospheric pressure:

surge | pressure ~ Normal(β₀ + β₁ × pressure, σ)

### Uncertainty Quantification

The Bayesian framework naturally provides:
- **Aleatory uncertainty**: Natural randomness in hazards
- **Epistemic uncertainty**: Parameter uncertainty from limited data
- **Structural uncertainty**: Model form uncertainty

Credible intervals and posterior predictive distributions quantify these uncertainties.

## 2. Agent-Based Model for Vulnerability

### Agent Representation

Each household agent i has:
- **Characteristics**: wealth, household size, education
- **Location**: (x, y) coordinates with elevation
- **State**: risk perception, adaptation status, damage

### Decision-Making Framework

Agents make adaptation decisions based on:

1. **Risk Perception Update**:
   ```
   R_i(t) = α × Experience + β × Education + γ × Social + δ × Exposure
   ```

2. **Adaptation Decision**:
   - Calculate NPV of adaptation:
     ```
     NPV = Σ_t [(Expected Damage) / (1 + r)^t] - Adaptation Cost
     ```
   - Adapt if: NPV > 0 AND Risk Perception > Threshold

3. **Damage Calculation**:
   ```
   Damage = Wealth × Depth-Damage Function × (1 - Adaptation Factor)
   ```

### Spatial Dynamics

Agents interact through:
- **Social learning**: Neighbors influence risk perception
- **Evacuation**: Movement during flood events
- **Resource competition**: Limited adaptation resources

### Vulnerability Assessment

Individual vulnerability is calculated as:

V = (Exposure × Sensitivity) / Adaptive Capacity

Where:
- **Exposure**: Based on elevation and hazard intensity
- **Sensitivity**: Function of wealth, household size
- **Adaptive Capacity**: Based on wealth, education, adaptation status

## 3. Deep Uncertainty Analysis

### Robust Decision Making (RDM)

Rather than optimizing for a single future, RDM identifies strategies that perform acceptably across many plausible futures.

#### Steps:

1. **Define Uncertainty Space**
   - Identify uncertain factors (SLR rate, storm frequency, socioeconomic trends)
   - Define plausible ranges for each factor

2. **Generate Scenarios**
   - Use Latin Hypercube Sampling for efficient coverage
   - Create N scenarios spanning uncertainty space

3. **Evaluate Strategies**
   - Define adaptation strategies (e.g., seawalls, nature-based solutions, retreat)
   - Simulate each strategy under all scenarios
   - Record performance on multiple objectives

4. **Assess Robustness**

   **Minimax Regret:**
   ```
   Regret_s(ω) = max_s' [Performance_s'(ω)] - Performance_s(ω)
   Robustness_s = -max_ω [Regret_s(ω)]
   ```

   **Percentile Performance:**
   ```
   Robustness_s = Performance at 10th percentile across scenarios
   ```

   **Satisficing:**
   ```
   Robustness_s = Fraction of scenarios meeting minimum thresholds
   ```

5. **Scenario Discovery**
   - Identify conditions where strategies fail
   - Use PRIM (Patient Rule Induction Method)
   - Find regions in uncertainty space associated with poor performance

### Adaptation Pathways

Dynamic strategies that change over time:

```
t=0 ──> Monitor ──> [Trigger 1] ──> Action A ──> [Trigger 2] ──> Action B
         │
         └──────> [Trigger 3] ──> Action C
```

Triggers based on:
- Observed sea level
- Flood frequency
- Economic indicators

## 4. Synthetic Event Generation

For data-sparse regions, we generate synthetic compound events:

### Physics-Based Generation

1. **Individual Hazards**:
   - Storm surge: Parametric cyclone model
   - Sea level: Trend + seasonal + stochastic components
   - Precipitation: Stochastic rainfall generator

2. **Dependency Structure**:
   - Use copulas to model joint distribution
   - Preserve observed correlations
   - Account for seasonal patterns

3. **Validation**:
   - Compare statistical properties to available observations
   - Check physical consistency (e.g., surge-pressure relationships)
   - Expert review of extreme events

### Uncertainty Propagation

Uncertainties in synthetic events propagate through:

```
Hazard Uncertainty ──> Network ──> Scenario Uncertainty ──> ABM ──> Impact Uncertainty
```

Quantified using:
- Monte Carlo simulation
- Sensitivity analysis (Sobol indices)
- Ensemble methods

## 5. Integration Workflow

### Complete Analysis Pipeline:

1. **Data Processing**
   - Load historical hazard data
   - Quality control and gap filling
   - Extract vulnerability data

2. **Bayesian Network Training**
   - Learn network structure
   - Estimate parameters with uncertainty
   - Validate against held-out data

3. **Scenario Generation**
   - Sample from posterior predictive
   - Apply future projections
   - Generate compound events

4. **Agent-Based Simulation**
   - Initialize household agents
   - Load spatial environment
   - Run for each scenario

5. **Uncertainty Analysis**
   - Define adaptation strategies
   - Evaluate across scenarios
   - Identify robust strategies
   - Perform scenario discovery

6. **Decision Support**
   - Visualize trade-offs
   - Create adaptation pathways
   - Quantify decision-relevant metrics

## Validation Framework

### Model Validation:

1. **Historical Validation**
   - Hindcast known events
   - Compare predicted vs. observed impacts
   - Metrics: RMSE, bias, skill scores

2. **Cross-Validation**
   - Leave-one-event-out for extreme events
   - Spatial cross-validation across cities
   - Temporal cross-validation

3. **Expert Evaluation**
   - Review by coastal engineers
   - Stakeholder feedback on plausibility
   - Comparison with local knowledge

### Uncertainty Validation:

- **Calibration**: Check if 95% credible intervals contain 95% of observations
- **Sharpness**: Evaluate precision of predictions
- **Coverage**: Assess if ensemble captures full range

## Key Innovations

1. **Data-Adaptive Hierarchy**: Automatically adjusts to data availability
2. **Compound Hazard Integration**: Explicit modeling of dependencies
3. **Behavioral Realism**: Agent-based approach captures adaptation dynamics
4. **Deep Uncertainty**: Robust strategies rather than optimal for single future
5. **Operational Focus**: Designed for real-world decision support

## Limitations and Future Work

### Current Limitations:
- Simplified physics in ABM
- Limited social network structure
- Computational cost for large cities

### Future Enhancements:
- Integration with full hydrodynamic models
- Machine learning for pattern recognition
- Real-time updating with new data
- Expansion to ecosystem impacts
