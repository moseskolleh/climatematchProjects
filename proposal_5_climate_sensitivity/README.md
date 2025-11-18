# Proposal 5: Multi-Constraint Framework for Climate Sensitivity

## Overview

This research develops a multi-constraint framework that combines paleoclimate, observational, and process-based constraints on climate sensitivity while explicitly accounting for structural uncertainties and potential non-analogues between past and future climate states.

## Scientific Background

**Equilibrium Climate Sensitivity (ECS)** is defined as the long-term global mean surface temperature change following a doubling of atmospheric CO₂ concentration. Despite decades of research, ECS remains one of the most uncertain quantities in climate science, with IPCC AR6 estimating a likely range of 2.5-4.0°C.

This uncertainty stems from:
- **Cloud feedbacks**: Different model representations of cloud processes
- **Aerosol forcing**: Historical aerosol effects remain poorly constrained
- **Pattern effects**: Regional warming patterns influence global feedback strength
- **State-dependence**: Climate sensitivity may vary with background climate state

## Research Objectives

1. **Develop a multi-line-of-evidence framework** that combines independent constraints from:
   - Paleoclimate proxies (Last Glacial Maximum, mid-Pliocene, Last Interglacial)
   - Historical observations (surface temperature, ocean heat content)
   - Process understanding (cloud feedbacks, regional patterns)

2. **Quantify and reduce structural uncertainties** through:
   - Explicit modeling of inter-constraint dependencies
   - Assessment of constraint applicability across climate states
   - Identification of common biases across model ensembles

3. **Provide actionable uncertainty estimates** for climate risk assessment

## Conservative Approach

Our framework prioritizes robustness through:

### 1. Multiple Independent Constraints
- Report uncertainty reduction only when multiple independent lines of evidence agree
- Explicitly test constraint independence using information theory metrics
- Use Bayesian model selection to weight constraints by their predictive skill

### 2. Explicit Structural Uncertainty
- Account for shared model biases using hierarchical Bayesian frameworks
- Distinguish between parametric and structural model uncertainties
- Quantify the "unknown unknowns" through discrepancy terms

### 3. State-Dependence and Pattern Effects
- Recognize that effective climate sensitivity varies with:
  - Background climate state (ice age vs. warm climate)
  - Forcing agent (CO₂ vs. aerosols vs. solar)
  - Timescale (transient vs. equilibrium response)
- Model these dependencies explicitly rather than assuming stationarity

### 4. Transparent Assumptions
- Document all methodological choices and their sensitivity
- Provide traceable uncertainty provenance
- Open-source all code and intermediate results

## Methodology

### 1. Constraint Integration Framework

#### Paleoclimate Constraints
```
ECS_paleo = ΔT_paleo / (ΔF_paleo / (3.7 W/m²))
```
Where:
- ΔT_paleo: Temperature change from proxy reconstructions
- ΔF_paleo: Radiative forcing change (estimated from ice cores, etc.)

**Data Sources:**
- Last Glacial Maximum (LGM): PMIP4 model ensemble + proxy compilations
- Mid-Pliocene Warm Period (mPWP): PlioMIP2 + proxy data
- Last Interglacial: Temperature and sea level reconstructions

**Key Challenges:**
- State-dependence: LGM may not be analogous to future warming
- Forcing uncertainty: Non-CO₂ forcings (ice sheets, vegetation) are uncertain
- Proxy uncertainty: Spatial coverage and dating errors

#### Observational Constraints
```
ECS_obs = ΔT_hist / (ΔF_hist / λ_eq)
```
Using:
- Historical surface temperature trends (1850-2020)
- Ocean heat uptake rates
- Top-of-atmosphere energy imbalance

**Emergent Constraints:**
- Correlate observable metrics (e.g., tropical cloud feedback, inter-annual variability) with ECS across models
- Apply observed values to constrain ECS probability distribution

#### Process-Based Constraints
- Cloud feedback decomposition (low-cloud, high-cloud, altitude changes)
- Lapse-rate and water vapor feedbacks
- Regional pattern scaling

### 2. Bayesian Integration

We use a hierarchical Bayesian framework:

```
p(ECS | D₁, D₂, ..., Dₙ) ∝ p(ECS) ∏ᵢ p(Dᵢ | ECS, θᵢ) p(θᵢ)
```

Where:
- D₁, D₂, ..., Dₙ: Different constraint datasets
- θᵢ: Nuisance parameters specific to each constraint
- p(ECS): Prior distribution (weakly informative)

**Handling Dependencies:**
- Model inter-constraint correlations explicitly
- Use copula methods for non-linear dependencies
- Test independence assumptions with synthetic data

### 3. Robust Validation Framework

#### Perfect Model Experiments
- Use one GCM as "truth", apply constraints from other models
- Test if true ECS is recovered within predicted uncertainty bounds
- Identify systematic biases in constraint methodology

#### Cross-Generation Consistency
- Apply CMIP5 constraints to predict CMIP6 ECS
- Evaluate whether constraint relationships are stable across model generations
- Assess risk of overfitting to current model ensembles

#### Constraint Independence Assessment
- Calculate mutual information between constraints
- Perform Principal Component Analysis on constraint space
- Weight constraints by their unique information content

### 4. Uncertainty Quantification

**Aleatory (Irreducible) Uncertainty:**
- Natural variability in observations
- Measurement errors in proxies

**Epistemic (Reducible) Uncertainty:**
- Model structural uncertainty
- Parameter uncertainty
- Incomplete process understanding

**Deep Uncertainty:**
- Unknown climate thresholds or tipping points
- Potential non-stationarity in climate feedbacks
- Emergent behaviors not captured by current models

## Implementation Plan

### Year 1: Data Compilation and Framework Development
- **Q1-Q2**: Compile and quality-control constraint datasets
  - Paleoclimate: PMIP4 model output + proxy compilations
  - Observations: HadCRUT5, CERES, Argo floats
  - Process: CMIP6 feedback decomposition

- **Q3-Q4**: Develop Bayesian integration framework
  - Implement hierarchical model in PyMC3/Stan
  - Code perfect model validation infrastructure
  - Establish baseline with existing methods (Lewis & Curry 2018, Sherwood et al. 2020)

### Year 2: Constraint Application and Validation
- **Q1-Q2**: Apply individual constraints
  - Estimate ECS from each line of evidence independently
  - Quantify constraint-specific uncertainties
  - Assess state-dependence and pattern effects

- **Q3-Q4**: Integrated constraint analysis
  - Combine constraints using Bayesian framework
  - Test independence assumptions
  - Perform sensitivity analyses on methodological choices

### Year 3: Robustness Testing and Refinement
- **Q1-Q2**: Perfect model experiments
  - CMIP5 → CMIP6 validation
  - Leave-one-model-out cross-validation
  - Synthetic data experiments

- **Q3-Q4**: Structural uncertainty assessment
  - Model discrepancy analysis
  - Shared bias identification
  - Constraint weighting optimization

### Year 4: Finalization and Dissemination
- **Q1-Q2**: Final ECS estimate with comprehensive uncertainty
  - Full posterior distribution
  - Scenario-dependent projections
  - Policy-relevant risk metrics

- **Q3-Q4**: Community engagement and knowledge transfer
  - Publications in Nature Climate Change / GRL
  - Code release and documentation
  - Workshops with climate modeling centers and IPCC authors

## Expected Outcomes

### Scientific Deliverables
1. **Refined ECS estimate** with:
   - Full posterior probability distribution
   - Explicit uncertainty provenance
   - State-dependent sensitivity estimates

2. **Methodological advances**:
   - Framework for combining heterogeneous constraints
   - Validation protocol for emergent constraints
   - Open-source toolkit for constraint-based inference

3. **Physical insights**:
   - Identification of dominant sources of uncertainty
   - Quantification of state-dependence in climate feedbacks
   - Assessment of constraint reliability across climate states

### Practical Applications
- **Climate policy**: More reliable warming projections for different emissions scenarios
- **Risk assessment**: Better characterized tail risks (low-probability high-impact outcomes)
- **Model development**: Targeted diagnostics for improving climate models

## Key Innovations

1. **Explicit state-dependence modeling**: Unlike previous studies that assume constant ECS, we model how sensitivity varies across climate states

2. **Comprehensive validation**: Perfect model experiments and cross-generation tests provide empirical evidence of constraint reliability

3. **Uncertainty provenance**: Transparent accounting of where uncertainties come from and whether they can be reduced

4. **Conservative reporting**: We prioritize robustness over precision, clearly communicating assumptions and limitations

## Data Requirements

### Input Data

| Data Type | Source | Spatial Resolution | Temporal Coverage |
|-----------|--------|-------------------|-------------------|
| Historical Temperature | HadCRUT5, BEST | Global gridded | 1850-2023 |
| Ocean Heat Content | Argo, WOD | 1°×1° | 2005-2023 |
| TOA Radiation | CERES EBAF | 1°×1° | 2000-2023 |
| LGM Temperature | PMIP4 + proxies | Regional | 21 ka |
| LGM Forcing | Ice core + model | Global | 21 ka |
| Pliocene Data | PlioMIP2 + proxies | Regional | 3 Ma |
| CMIP6 Models | ESGF | Native grid | Historical + scenarios |

### Processing Requirements
- **Storage**: ~2 TB for CMIP6 subset + observations
- **Compute**: Bayesian MCMC (10⁴ - 10⁵ samples) requires HPC access
- **Software**: Python (PyMC3, xarray), R (INLA), Julia (Turing.jl)

## Validation Metrics

1. **Constraint Consistency**: Do multiple constraints agree within uncertainties?
2. **Cross-Validation Skill**: Perfect model test success rate
3. **Posterior Coverage**: Fraction of true values within predicted intervals
4. **Information Gain**: KL divergence between prior and posterior
5. **Constraint Independence**: Mutual information between constraint pairs

## Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Constraints systematically biased | Medium | High | Perfect model validation, ensemble diversity |
| State-dependence invalidates constraints | Medium | High | Explicit modeling, sensitivity tests |
| Insufficient data for rare climate states | High | Medium | Synthetic data augmentation, wider priors |
| Computational bottlenecks | Low | Medium | Scalable algorithms, HPC access |
| Non-stationarity in climate feedbacks | Low | High | Multiple time period analysis, mechanistic understanding |

## Team and Resources

### Personnel
- **Principal Investigator** (20%): Project oversight, stakeholder engagement
- **Postdoc 1** (100%): Paleoclimate constraints, state-dependence analysis
- **Postdoc 2** (100%): Observational constraints, validation framework
- **PhD Student** (100%): Process-based constraints, Bayesian integration
- **Research Engineer** (50%): Data pipeline, computational infrastructure

### Computational Resources
- Local workstation for development
- HPC allocation: 100k CPU hours/year for MCMC sampling
- Cloud storage: 5 TB for CMIP6 data

### Budget (4 years)
- **Personnel**: €800,000 (salaries + benefits)
- **Computational**: €50,000 (HPC time, cloud storage)
- **Travel**: €30,000 (conferences, collaborations)
- **Publications**: €10,000 (open access fees)
- **Workshops**: €20,000 (stakeholder engagement)
- **Total**: €910,000

## Stakeholder Engagement

### Scientific Community
- **CMIP modeling centers**: Provide feedback on constraint validity
- **IPCC authors**: Ensure methods align with assessment needs
- **Paleoclimate community**: Co-develop proxy-model comparison frameworks

### Policy Relevance
- **UNFCCC**: Inform NDC ambition and climate finance discussions
- **National climate services**: Provide uncertainty guidance for adaptation planning
- **Financial sector**: Support climate risk disclosure (TCFD framework)

## Publications Strategy

### Primary Papers (Target: Nature Climate Change, Science Advances)
1. "A Multi-Constraint Framework for Climate Sensitivity" (Year 2)
2. "State-Dependence in Climate Feedbacks from Paleo to Future" (Year 3)
3. "Robust Validation of Emergent Constraints on Climate Sensitivity" (Year 3)

### Methods Papers (Target: GMD, JAMES)
4. "Bayesian Integration of Heterogeneous Climate Constraints" (Year 2)
5. "Perfect Model Validation for Emergent Constraints" (Year 3)

### Data Papers
6. "A Curated Multi-Line-of-Evidence Dataset for ECS Constraints" (Year 2)

## Code and Data Availability

All code will be released under MIT license on GitHub:
- `climate-sensitivity-constraints/` repository
- Comprehensive documentation and tutorials
- Reproducible workflows using Docker containers
- Archived versions on Zenodo with DOIs

Processed datasets will be published on Zenodo with full metadata.

## Repository Structure

```
proposal_5_climate_sensitivity/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── config/                   # Configuration files
│   ├── data_sources.yml     # Data source specifications
│   ├── model_config.yml     # Bayesian model configuration
│   └── validation_config.yml # Validation parameters
├── data/                     # Data directory (not in git)
│   ├── raw/                 # Original data files
│   ├── processed/           # Cleaned and standardized data
│   └── interim/             # Intermediate processing steps
├── src/                      # Source code
│   ├── constraints/         # Constraint implementations
│   │   ├── paleoclimate.py  # LGM, Pliocene, etc.
│   │   ├── observational.py # Historical, energy budget
│   │   └── process_based.py # Cloud feedbacks, etc.
│   ├── uncertainty/         # Uncertainty quantification
│   │   ├── bayesian_integration.py
│   │   ├── dependency_modeling.py
│   │   └── uncertainty_decomposition.py
│   ├── validation/          # Validation framework
│   │   ├── perfect_model.py
│   │   ├── cross_validation.py
│   │   └── constraint_independence.py
│   ├── models/              # Climate model interfaces
│   │   ├── cmip6_handler.py
│   │   └── feedback_decomposition.py
│   ├── data_processing/     # Data ingestion and QC
│   │   ├── observations.py
│   │   ├── proxies.py
│   │   └── model_output.py
│   └── utils/               # Utility functions
│       ├── plotting.py
│       ├── statistics.py
│       └── io_helpers.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_individual_constraints.ipynb
│   ├── 03_bayesian_integration.ipynb
│   ├── 04_validation_analysis.ipynb
│   └── 05_results_visualization.ipynb
├── tests/                   # Unit tests
│   ├── test_constraints.py
│   ├── test_uncertainty.py
│   └── test_validation.py
├── docs/                    # Documentation
│   ├── methodology.md       # Detailed methods
│   ├── data_requirements.md # Data specifications
│   ├── validation_protocol.md # Validation procedures
│   └── api_reference.md     # Code documentation
└── results/                 # Output directory
    ├── figures/
    ├── tables/
    └── reports/
```

## Contact

**Moses Kolleh**
Environmental Scientist & Sustainable AI Researcher
Digital Society School, Amsterdam University of Applied Sciences

## References

- Sherwood, S. C., et al. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. *Reviews of Geophysics*, 58(4), e2019RG000678.
- IPCC (2021). Climate Change 2021: The Physical Science Basis. *Sixth Assessment Report*.
- Mauritsen, T., & Pincus, R. (2017). Committed warming inferred from observations. *Nature Climate Change*, 7(9), 652-655.
- Annan, J. D., & Hargreaves, J. C. (2020). Partitioning uncertainty in climate sensitivity. *Earth System Dynamics*, 11(4), 1053-1068.
- Lewis, N., & Curry, J. (2018). The impact of recent forcing and ocean heat uptake data on estimates of climate sensitivity. *Journal of Climate*, 31(15), 6051-6071.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research is conducted as part of the Enhanced Climate Science Research Proposals initiative, addressing data challenges and operational feasibility in African climate science.

---

*Last updated: November 2025*
