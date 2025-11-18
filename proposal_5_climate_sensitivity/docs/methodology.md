# Methodology

## Overview

This document describes the methodology for constraining equilibrium climate sensitivity (ECS) using multiple lines of evidence within a Bayesian framework.

## Climate Sensitivity Definition

**Equilibrium Climate Sensitivity (ECS)** is defined as:

> The change in global mean surface temperature (GMST) following a doubling of atmospheric CO₂ concentration, after the climate system has reached a new equilibrium (with sea surface temperatures and ocean heat content adjusted, but ice sheets, vegetation, and other slow components held fixed).

Mathematically:
```
ECS = ΔT_eq / (ΔF_2xCO2 / λ)
```

where:
- ΔT_eq: Equilibrium temperature change
- ΔF_2xCO2: Radiative forcing from CO₂ doubling (~3.7 W/m²)
- λ: Climate feedback parameter (W/m²/K)

## Constraint Approaches

### 1. Paleoclimate Constraints

#### 1.1 Last Glacial Maximum (LGM)

The LGM (~21 ka) provides a constraint from a colder climate state:

**Method:**
```
ECS = ΔT_LGM / (ΔF_LGM / F_2xCO2)
```

**Data Sources:**
- Temperature: Proxy compilations (Tierney et al. 2020)
- Forcing: PMIP4 models + ice core data

**State-Dependence Correction:**
Climate sensitivity may differ between cold (LGM) and warm (future) states due to:
- Different cloud feedback regimes
- Ice-albedo feedback saturation
- Changes in atmospheric circulation

We apply a correction factor of ~1.1 to adjust LGM-derived ECS to future warming context.

#### 1.2 Mid-Pliocene Warm Period (mPWP)

The mPWP (~3 Ma) was 2-4K warmer with CO₂ ~400 ppm:

**Advantages:**
- Better analogue for future warming
- No large ice sheets complicating forcing
- Well-constrained CO₂ levels

**Method:**
```
ECS = ΔT_Plio / (ΔF_Plio / F_2xCO2)
ΔF_Plio = 5.35 * ln(CO2_Plio / CO2_PI) + F_other
```

#### 1.3 Uncertainty Propagation

All paleoclimate constraints use Monte Carlo sampling to propagate uncertainties:
1. Sample temperature and forcing from joint distribution
2. Account for correlations (typically negative)
3. Calculate ECS distribution
4. Filter unphysical values (ECS < 0 or > 15 K)

### 2. Observational Constraints

#### 2.1 Energy Budget Method

Uses historical warming (1850-2020) and energy budget:

**Method:**
```
ECS = (ΔF_hist - ΔQ) / ΔT_hist * F_2xCO2
```

where:
- ΔT_hist: Observed warming (~1.1 K)
- ΔF_hist: Effective radiative forcing (~2.7 W/m²)
- ΔQ: Ocean heat uptake rate (~0.6 W/m²)

**Data Sources:**
- Temperature: HadCRUT5, Berkeley Earth
- Forcing: IPCC AR6 synthesis
- Ocean heat: Argo floats, CERES

**Challenges:**
- Pattern effects: Regional warming patterns affect feedback strength
- Aerosol uncertainty: Historical aerosol forcing is poorly constrained
- Internal variability: Observed trend includes natural variations

#### 2.2 Emergent Constraints

Exploit correlations between observable quantities and ECS across climate models:

**Method:**
1. Identify predictor X that correlates with ECS in models
2. Fit regression: ECS = a + b * X
3. Use observed X_obs to constrain ECS
4. Propagate uncertainties (observation + regression)

**Examples:**
- Tropical cloud variability
- Lower tropospheric mixing
- Southern Ocean warming pattern

**Validation:**
- Cross-validation to prevent overfitting
- Physical understanding required
- Test independence across model generations

### 3. Process-Based Constraints

#### 3.1 Feedback Decomposition

Climate feedback parameter:
```
λ = λ_planck + λ_wv + λ_lr + λ_cloud + λ_albedo
```

Component feedbacks:
- **Planck**: -3.2 W/m²/K (direct radiative response)
- **Water vapor**: +1.8 W/m²/K (moistening amplifies)
- **Lapse rate**: -0.8 W/m²/K (upper troposphere warms faster)
- **Cloud**: +0.4 W/m²/K (largest uncertainty!)
- **Albedo**: +0.4 W/m²/K (ice-albedo feedback)

**Method:**
1. Estimate each feedback from observations/theory
2. Account for correlations (e.g., water vapor-lapse rate)
3. Sum feedbacks: λ_total = Σ λ_i
4. Calculate: ECS = F_2xCO2 / (-λ_total)

#### 3.2 Cloud Feedback Constraint

Cloud feedbacks decomposed into:
- **Low clouds**: Subtropical stratocumulus regions
- **High clouds**: Tropical anvil altitude changes
- **Middle clouds**: Moderate uncertainty

Constrained using:
- Satellite observations (CERES, ISCCP)
- Process-resolving models
- Large-eddy simulations

## Bayesian Integration

### Framework

Combine constraints using hierarchical Bayesian model:

```
p(ECS | D₁, D₂, ..., Dₙ) ∝ p(ECS) ∏ᵢ p(Dᵢ | ECS, θᵢ) p(θᵢ)
```

where:
- p(ECS): Prior distribution
- D₁, D₂, ..., Dₙ: Constraint datasets
- θᵢ: Nuisance parameters (uncertainties, biases)

### Prior Choices

**Jeffreys Prior (default):**
```
p(ECS) ∝ 1/ECS
```
- Scale-invariant
- Minimally informative
- Avoids biasing toward high values

**Informed Prior (alternative):**
```
p(ECS) ~ Normal(μ=3.0, σ=1.5)
```
- Based on CMIP6 ensemble
- Weakly informative
- Useful for sensitivity tests

### Handling Dependencies

Constraints may not be independent due to:
- Shared observational data
- Common model biases
- Physical connections

**Approach:**
1. Estimate correlation matrix between constraints
2. Adjust likelihood contributions
3. Calculate effective degrees of freedom
4. Downweight highly correlated constraints

### Uncertainty Quantification

**Aleatory (irreducible) uncertainty:**
- Natural climate variability
- Measurement errors

**Epistemic (reducible) uncertainty:**
- Model structural uncertainty
- Parameter uncertainty
- Incomplete process understanding

**Output:**
- Full posterior distribution p(ECS | data)
- Percentiles: 5th, 16th, 50th, 84th, 95th
- Information gain (KL divergence from prior)

## Validation

### Perfect Model Experiments

Test constraint methodology:

1. **Leave-one-out:** Use model M as "truth", apply constraints from other models
2. **Check coverage:** Is true ECS_M within predicted uncertainty bounds?
3. **Calculate bias:** Mean(estimated - true)
4. **Assess calibration:** Are confidence intervals well-calibrated?

**Metrics:**
- Coverage rate (target: 90% for 90% CI)
- Mean absolute error
- Root mean square error
- Calibration statistics (Kolmogorov-Smirnov test)

### Cross-Generation Validation

Test stability across model generations:

1. Apply CMIP5 constraints to predict CMIP6 ECS
2. Compare predicted vs actual CMIP6 ensemble
3. Assess whether relationships are stable

**Purpose:**
- Guard against overfitting to single model ensemble
- Test robustness of emergent constraints

### Independence Assessment

Test whether constraints provide unique information:

**Methods:**
- Mutual information calculation
- Principal component analysis
- Conditional independence tests

**Action:**
- Weight constraints by unique information content
- Avoid double-counting shared information

## Uncertainty Decomposition

Break down total uncertainty:

```
Var(ECS) = Var_paleo + Var_obs + Var_process + Cov(constraints)
```

**Information gain:**
```
IG = H(prior) - H(posterior)
```

Quantifies how much uncertainty is reduced by constraints.

## Sensitivity Analysis

Test robustness to methodological choices:

**Varied parameters:**
- Prior type and parameters
- Constraint weights
- Correlation assumptions
- Outlier treatment
- State-dependence corrections

**Report:**
- Parameter sensitivity indices
- Conditional distributions
- Scenario comparisons

## Computational Implementation

**Tools:**
- PyMC3: Bayesian inference
- NumPy/SciPy: Numerical computations
- xarray: Climate data handling

**Performance:**
- MCMC sampling: ~50,000 samples, 4 chains
- Convergence: Gelman-Rubin R̂ < 1.01
- Runtime: ~1-2 hours on modern workstation

## References

1. Sherwood, S. C., et al. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. *Reviews of Geophysics*, 58(4).

2. Annan, J. D., & Hargreaves, J. C. (2020). Partitioning uncertainty in climate sensitivity. *Earth System Dynamics*, 11(4).

3. Tierney, J. E., et al. (2020). Glacial cooling and climate sensitivity revisited. *Nature*, 584(7822).

4. Lewis, N., & Curry, J. (2018). The impact of recent forcing and ocean heat uptake data on estimates of climate sensitivity. *Journal of Climate*, 31(15).

5. IPCC (2021). Climate Change 2021: The Physical Science Basis. Chapter 7: The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity.
