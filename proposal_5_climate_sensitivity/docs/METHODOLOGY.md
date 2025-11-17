# Methodology

## Overview

This document details the scientific methodology behind the multi-constraint framework for estimating equilibrium climate sensitivity (ECS).

## 1. Equilibrium Climate Sensitivity (ECS)

ECS is defined as the equilibrium change in global mean surface temperature following a doubling of atmospheric CO2 concentration.

### 1.1 Definition

```
ECS = ΔT_eq when CO2 doubles from pre-industrial level (280 ppm → 560 ppm)
```

### 1.2 Relationship to Climate Feedback

```
ECS = -ΔF_2x / λ
```

Where:
- `ΔF_2x` = radiative forcing from CO2 doubling (≈ 3.7 W/m²)
- `λ` = climate feedback parameter (W/m²/K)

## 2. Constraint Framework

### 2.1 Bayesian Framework

We use Bayes' theorem to update our prior distribution of ECS based on constraints:

```
P(ECS | constraints) ∝ P(constraints | ECS) × P(ECS)
```

Where:
- `P(ECS)` = prior distribution from CMIP models
- `P(constraints | ECS)` = likelihood of observing constraints given ECS
- `P(ECS | constraints)` = posterior distribution

### 2.2 Multi-Constraint Combination

For independent constraints, we multiply likelihoods:

```
P(ECS | C1, C2, ..., Cn) ∝ P(C1 | ECS) × P(C2 | ECS) × ... × P(Cn | ECS) × P(ECS)
```

## 3. Individual Constraints

### 3.1 Paleoclimate Constraints

#### Last Glacial Maximum (LGM)

**Observable:** Global mean temperature change (ΔT_LGM ≈ -5°C)

**Relationship to ECS:**
```
ΔT_LGM = (ΔF_LGM / ΔF_2x) × ECS × f_state
```

Where:
- `ΔF_LGM` = total LGM forcing (CO2 + ice sheets + vegetation)
- `f_state` = state-dependence correction factor (≈ 0.9)

**Implementation:**
1. Calculate expected LGM cooling for each model
2. Compare to proxy reconstructions
3. Weight models by fit quality

#### Mid-Pliocene Warm Period (MPWP)

**Observable:** Global warming (ΔT_MPWP ≈ 2.5°C)

**Advantage:** CO2 similar to modern (less state-dependence)

### 3.2 Observational Constraints

#### Historical Warming

**Observable:** Observed warming 1850-2020 (ΔT_obs ≈ 1.1°C)

**Relationship to ECS:**
```
ΔT_obs = f_realized × (ΔF_hist / ΔF_2x) × ECS
```

Where:
- `ΔF_hist` = historical net forcing
- `f_realized` = realized warming fraction (≈ 0.65, accounts for ocean heat uptake)

**Key challenges:**
- Aerosol forcing uncertainty
- Internal variability
- Incomplete equilibration

#### Energy Budget

**Observable:** TOA radiation imbalance (N ≈ 0.7 W/m²)

**Energy balance:**
```
N = ΔF - λ × ΔT
λ = ΔF_2x / ECS
```

### 3.3 Process-Based Constraints

#### Cloud Feedback

Uses emergent constraints relating present-day cloud properties to future cloud feedbacks.

#### Water Vapor and Lapse Rate

Based on Clausius-Clapeyron scaling and observed vertical temperature structure.

## 4. Uncertainty Quantification

### 4.1 Sources of Uncertainty

1. **Observational uncertainty:** Measurement errors, spatial coverage
2. **Model structural uncertainty:** Different parameterizations
3. **Internal variability:** Chaotic climate fluctuations
4. **Forcing uncertainty:** Historical aerosols, etc.
5. **State-dependence:** Sensitivity varying with climate state

### 4.2 Uncertainty Propagation

- Monte Carlo sampling for all uncertain inputs
- Bayesian inference for parameter estimation
- Ensemble methods for model structural uncertainty

### 4.3 Uncertainty Decomposition

Total variance decomposed into:

```
Var(total) = Var(within-constraint) + Var(between-constraint)
```

## 5. Validation

### 5.1 Perfect Model Tests

**Procedure:**
1. Select one model as "truth"
2. Generate pseudo-observations from that model
3. Apply constraint using remaining models
4. Check if constraint recovers true ECS

**Metrics:**
- Bias: difference between estimated and true ECS
- RMSE: root mean square error
- Coverage: fraction of times true value falls within credible interval

### 5.2 Independence Testing

Test correlation between model weights from different constraints.

**Criterion:** Constraints considered independent if correlation < 0.7

## 6. Conservative Reporting Rules

1. **Multiple constraint agreement:** Only report narrowed uncertainty if ≥2 independent constraints agree

2. **Transparent assumptions:** Explicitly state all assumptions about:
   - Constraint independence
   - State-dependence corrections
   - Forcing estimates

3. **Structural uncertainty:** Report and account for model structural differences

4. **Limitations:** Clearly document known limitations and caveats

## 7. Comparison with IPCC AR6

IPCC AR6 assessed ECS as "likely" (>66% probability) between 2.5°C and 4.0°C, with a best estimate of 3.0°C.

Our framework aims to:
- Provide explicit methodology for combining constraints
- Quantify and report all uncertainties
- Enable reproducibility through open-source implementation

## References

1. Sherwood, S. C., et al. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. *Reviews of Geophysics*, 58(4).

2. Tierney, J. E., et al. (2020). Past climates inform our future. *Science*, 370(6517), eaay3701.

3. IPCC (2021). Climate Change 2021: The Physical Science Basis. Chapter 7: The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity.

4. Zhu, J., & Poulsen, C. J. (2021). On the increase of climate sensitivity in the past. *Nature Geoscience*, 14(7), 463-468.
