# Proposal 3: Theory-Guided Discovery of Climate System Connections

## Overview

This research project combines machine learning with dynamical systems theory to discover physically-meaningful teleconnections in the climate system. By constraining pattern search with theoretical understanding of atmospheric dynamics and implementing rigorous statistical validation, we identify robust climate connections while controlling false discovery rates.

## Project Status

**Current Phase**: Initial Setup
**Start Date**: November 2025
**Duration**: 4 years
**Institution**: Digital Society School, Amsterdam University of Applied Sciences

## Research Objectives

### Primary Goals

1. **Theory-Constrained Discovery**: Develop machine learning methods that respect physical constraints from atmospheric dynamics
2. **Robust Statistical Validation**: Implement rigorous validation frameworks with false discovery rate control
3. **Physical Interpretation**: Establish protocols for mechanistic understanding of discovered connections
4. **Operational Implementation**: Create tools for climate prediction enhancement in African contexts

### Key Innovations

- **Dynamical Guidance**: Search constrained to patterns consistent with Rossby wave propagation, conserved quantities, and known timescales
- **Causal Hierarchy**: Distinguish between direct causation, mediated connections, and common drivers
- **Hypothesis-Driven Approach**: Generate candidates based on dynamical theory, not purely data-driven search
- **Ensemble Validation**: Test discoveries across multiple reanalysis products

## Scientific Approach

### 1. Theory-Constrained Discovery Framework

Our approach integrates physical understanding at every stage:

- **Rossby Wave Dynamics**: Constrain spatial patterns to be consistent with wave propagation paths
- **Conservation Laws**: Enforce consistency with potential vorticity and energy conservation
- **Timescale Matching**: Search only at physically-relevant temporal scales
- **Stratospheric-Tropospheric Coupling**: Include vertical structure in pattern detection

### 2. Causal Discovery Methods

We employ multiple complementary approaches:

- **Granger Causality**: Time-lagged correlations with statistical significance testing
- **Convergent Cross Mapping**: Detect nonlinear dynamical coupling
- **Transfer Entropy**: Quantify directed information flow
- **Pearl's Causal Framework**: Structural equation modeling with physical constraints

### 3. Statistical Validation Pipeline

- Bootstrap confidence intervals for all discoveries
- False Discovery Rate (FDR) control using Benjamini-Hochberg procedure
- Out-of-sample testing on independent time periods (training: 1979-2000, testing: 2001-2020)
- Cross-validation across multiple reanalysis datasets (ERA5, MERRA-2, JRA-55)
- Sensitivity analysis to methodological choices

## Methodology

### Data Sources

**Primary Datasets**:
- ERA5 Reanalysis (1979-present)
- MERRA-2 (1980-present)
- JRA-55 (1958-present)

**Variables**:
- Geopotential height (multiple levels)
- Sea surface temperature
- Outgoing longwave radiation
- Precipitation
- Wind fields (u, v components)

**Spatial Focus**: Global with emphasis on connections affecting African climate

### Analytical Workflow

```
1. Data Preprocessing
   ├── Quality control and gap filling
   ├── Detrending and deseasonalization
   └── Spatial/temporal filtering

2. Candidate Generation
   ├── Theory-based hypothesis formation
   ├── Physical constraint definition
   └── Search space specification

3. Pattern Discovery
   ├── Apply causal discovery algorithms
   ├── Enforce physical constraints
   └── Identify candidate teleconnections

4. Statistical Validation
   ├── Bootstrap significance testing
   ├── FDR control
   ├── Cross-validation
   └── Robustness checks

5. Physical Interpretation
   ├── Composite analysis
   ├── Energy/momentum budgets
   ├── Climate model experiments
   └── Expert evaluation

6. Documentation and Dissemination
   ├── Scientific publications
   ├── Operational integration
   └── Stakeholder communication
```

## Project Structure

```
proposal_3_climate_connections/
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── methodology.md                 # Detailed methodology
│   ├── theoretical_framework.md       # Physical theory foundation
│   ├── validation_framework.md        # Statistical validation approach
│   ├── implementation_timeline.md     # 4-year roadmap
│   └── technical_architecture.md      # Software architecture
├── src/                               # Source code
│   ├── data_processing/              # Data acquisition and preprocessing
│   ├── discovery/                    # Teleconnection discovery algorithms
│   ├── validation/                   # Statistical validation tools
│   └── visualization/                # Results visualization
├── tests/                            # Unit and integration tests
├── config/                           # Configuration files
├── data/                             # Data directory
│   ├── raw/                         # Raw reanalysis data
│   ├── processed/                   # Preprocessed data
│   └── results/                     # Analysis results
├── notebooks/                        # Jupyter notebooks for analysis
└── scripts/                          # Utility scripts
```

## Key Components

### 1. Dynamical Constraint Engine
Enforces physical consistency through:
- Rossby wave dispersion relations
- Momentum and energy conservation
- Geostrophic/hydrostatic balance
- Potential vorticity constraints

### 2. Causal Discovery Module
Implements multiple methods:
- Vector Autoregression (VAR) with Granger causality
- Convergent Cross Mapping (CCM) for nonlinear systems
- Transfer entropy with appropriate lag selection
- Structural causal models with do-calculus

### 3. Validation Framework
Rigorous statistical testing:
- Multiple hypothesis correction (Benjamini-Hochberg FDR)
- Bootstrap resampling (10,000 iterations)
- Cross-validation across reanalysis products
- Perfect model experiments

### 4. Physical Interpretation Toolkit
Tools for mechanistic understanding:
- Composite analysis routines
- Budget diagnostics (energy, momentum, vorticity)
- Climate model perturbation experiments
- Expert elicitation frameworks

## Expected Outcomes

### Year 1: Foundation
- Data pipeline established
- Constraint framework implemented
- Baseline method comparisons completed
- Initial candidate teleconnections identified

### Year 2: Discovery
- Comprehensive teleconnection catalog for African climate
- Statistical validation completed
- Physical mechanisms characterized
- First publications submitted

### Year 3: Integration
- Integration with Proposals 1 & 2 (drought and flood prediction)
- Operational prototype developed
- Climate model validation studies
- Stakeholder engagement initiated

### Year 4: Operationalization
- Operational system deployed
- Technology transfer to African institutions
- Comprehensive documentation
- Final publications and synthesis

## Validation Metrics

### Statistical Performance
- **Significance Levels**: p < 0.01 after FDR correction
- **Effect Sizes**: Cohen's d > 0.5 for practically significant relationships
- **Reproducibility**: Agreement across ≥2 independent reanalysis products
- **Temporal Stability**: Consistent patterns in 2+ independent time periods

### Physical Consistency
- **Energy Conservation**: Residuals < 10% in budget analyses
- **Wave Dynamics**: Patterns consistent with Rossby wave theory
- **Model Agreement**: Relationships captured in ≥5 CMIP6 models
- **Expert Validation**: Agreement from ≥3 independent dynamical meteorologists

## Stakeholder Engagement

### Primary Stakeholders
- African meteorological services
- Regional climate centers (ICPAC, ACMAD)
- Research institutions (universities, SASSCAL)
- Operational forecasting agencies

### Engagement Activities
- Quarterly webinars on findings
- Annual workshops for capacity building
- Co-development of operational tools
- Training programs for local scientists

## Budget Overview

**Total**: €500,000 over 4 years

**Breakdown**:
- Personnel: €300,000 (2 PhD students, 1 postdoc)
- Computing Infrastructure: €100,000 (HPC access, cloud storage)
- Capacity Building: €50,000 (workshops, training materials)
- Travel & Conferences: €30,000
- Publication & Dissemination: €20,000

## Team

**Principal Investigator**: Moses Kolleh
**Affiliation**: Digital Society School, Amsterdam University of Applied Sciences
**Collaborators**:
- African meteorological services
- International climate research centers
- Dynamical meteorology experts

## References & Resources

### Key Literature
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Sugihara et al. (2012). Detecting Causality in Complex Ecosystems
- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models
- Wallace & Gutzler (1981). Teleconnections in the Geopotential Height Field

### Data Resources
- Copernicus Climate Data Store (ERA5)
- NASA GES DISC (MERRA-2)
- JMA (JRA-55)
- CMIP6 Model Archive

## Getting Started

See [docs/implementation_timeline.md](docs/implementation_timeline.md) for detailed implementation plan and [docs/technical_architecture.md](docs/technical_architecture.md) for software setup instructions.

## License

This research project is part of the ClimateMatch Projects initiative focused on advancing African climate science.

## Contact

For questions or collaboration opportunities:
- **Researcher**: Moses
- **Institution**: Digital Society School, Amsterdam University of Applied Sciences
- **Project Repository**: [GitHub - climatematchProjects](https://github.com/moseskolleh/climatematchProjects)

---

*Last Updated: November 2025*
