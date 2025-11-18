# ClimateMatch Projects: Enhanced Climate Science Research Proposals for Africa

A comprehensive portfolio of five interconnected research proposals addressing critical climate challenges in Africa through innovative integration of machine learning, physics-based modeling, and local knowledge systems.

**Principal Investigator**: Moses Kolleh
**Institution**: Digital Society School, Amsterdam University of Applied Sciences
**Focus Region**: Africa
**Program Duration**: 2025-2029 (4 years)
**Total Budget**: €2,640,000

## Overview

This research program tackles Africa's most pressing climate challenges through five complementary proposals that share a common philosophy: combining cutting-edge computational methods with explicit recognition of data limitations, deep uncertainty quantification, and meaningful stakeholder engagement.

### Core Principles

1. **Data-Adaptive Approaches**: Methods that work despite sparse observational networks
2. **Physics-Informed Machine Learning**: Combining physical understanding with data-driven discovery
3. **Uncertainty-Centric**: Explicit quantification and communication of uncertainties
4. **Co-Development**: Meaningful engagement with African institutions and communities
5. **Operational Feasibility**: Transition pathways to sustained local operation

## Research Proposals

### 1. Adaptive Multi-Scale Drought Early Warning System

**Directory**: [`proposal-01-drought-early-warning/`](./proposal-01-drought-early-warning/)
**Budget**: €650,000
**Duration**: 4 years

Develops an adaptive drought early warning system combining ensemble machine learning with local knowledge systems. Provides probabilistic forecasts at 1-6 month lead times through hierarchical Bayesian data fusion.

**Key Features**:
- Multi-scale prediction architecture (continental to community scale)
- Integration of satellite observations, climate forecasts, and indigenous indicators
- Dynamic baseline adaptation using online learning
- Explicit uncertainty quantification

**Target Impact**: Enhanced drought preparedness across East and Southern Africa

[Read More →](./proposal-01-drought-early-warning/README.md)

---

### 2. Hybrid Physics-ML Framework for Flood Prediction

**Directory**: [`proposal_2_flood_prediction/`](./proposal_2_flood_prediction/)
**Budget**: €580,000
**Duration**: 4 years

Develops a hybrid framework combining simplified physics-based models with machine learning for flood prediction in data-sparse West African rivers. Uses differentiable hydrological models within neural architectures.

**Key Features**:
- Differentiable physics core ensuring mass conservation
- Hierarchical transfer learning for ungauged basins
- Probabilistic forecasts with 24-72 hour lead times
- Integration with national disaster management systems

**Target Basins**: Niger, Volta, Senegal, Gambia, and Mono River Basins

[Read More →](./proposal_2_flood_prediction/README.md)

---

### 3. Theory-Guided Discovery of Climate System Connections

**Directory**: [`proposal_3_climate_connections/`](./proposal_3_climate_connections/)
**Budget**: €500,000
**Duration**: 4 years

Combines machine learning with dynamical systems theory to discover physically-meaningful teleconnections affecting African climate. Constrains pattern search with atmospheric dynamics theory while implementing rigorous statistical validation.

**Key Features**:
- Theory-constrained discovery respecting physical laws
- Multiple causal discovery methods (Granger causality, convergent cross mapping, transfer entropy)
- Rigorous validation with false discovery rate control
- Integration with Proposals 1 & 2 for improved predictions

**Focus**: Global teleconnections affecting African rainfall and temperature

[Read More →](./proposal_3_climate_connections/README.md)

---

### 4. Integrated Coastal Risk Framework for African Cities

**Directory**: [`proposal_4_coastal_risk/`](./proposal_4_coastal_risk/)
**Duration**: 4 years

Develops an integrated framework for compound coastal hazards combining Bayesian networks with agent-based models. Addresses data limitations through hierarchical modeling and synthetic event generation.

**Key Features**:
- Compound hazard integration (sea level rise, storm surge, flooding, erosion)
- Bayesian networks for hazard dependencies
- Agent-based modeling for household-level vulnerability
- Deep uncertainty methods for robust decision making

**Target Cities**: Lagos, Mombasa, Dakar, and Maputo

[Read More →](./proposal_4_coastal_risk/README.md)

---

### 5. Multi-Constraint Framework for Climate Sensitivity

**Directory**: [`proposal_5_climate_sensitivity/`](./proposal_5_climate_sensitivity/)
**Budget**: €910,000
**Duration**: 4 years

Develops a multi-constraint framework combining paleoclimate, observational, and process-based constraints on Equilibrium Climate Sensitivity (ECS). Explicitly accounts for structural uncertainties and state-dependence.

**Key Features**:
- Integration of multiple independent lines of evidence
- Hierarchical Bayesian framework for constraint combination
- Perfect model validation experiments
- Explicit modeling of state-dependence in climate feedbacks

**Global Significance**: Refined climate sensitivity estimates reduce uncertainty in future warming projections

[Read More →](./proposal_5_climate_sensitivity/README.md)

---

## Interconnections Between Proposals

The five proposals are designed to complement and reinforce each other:

```
┌─────────────────────────────────────────────────────────────────┐
│  Proposal 5: Climate Sensitivity                                │
│  Provides improved warming projections for all other proposals  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Proposal 1  │ │  Proposal 2  │ │  Proposal 4  │
│   Drought    │ │    Flood     │ │   Coastal    │
│   Warning    │ │  Prediction  │ │     Risk     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Proposal 3     │
              │ Teleconnections  │
              │ (Improves 1, 2)  │
              └──────────────────┘
```

**Specific Synergies**:
- **Proposal 3 → Proposals 1 & 2**: Discovered teleconnections improve drought and flood predictions
- **Proposal 5 → All**: Refined climate sensitivity informs future climate scenarios
- **Proposals 1, 2, 4**: Share methodological approaches (Bayesian frameworks, uncertainty quantification)
- **All**: Common focus on data-sparse environments and operational feasibility

## Repository Structure

```
climatematchProjects/
├── README.md                              # This file
├── LICENSE                                # Project license
├── improved_climate_proposals.docx.pdf    # Original proposal document
│
├── proposal-01-drought-early-warning/     # Proposal 1
│   ├── README.md
│   ├── docs/
│   ├── src/
│   ├── data/
│   ├── tests/
│   ├── stakeholders/
│   └── notebooks/
│
├── proposal_2_flood_prediction/           # Proposal 2
│   ├── README.md
│   ├── docs/
│   ├── src/
│   ├── data/
│   ├── notebooks/
│   ├── results/
│   └── config/
│
├── proposal_3_climate_connections/        # Proposal 3
│   ├── README.md
│   ├── docs/
│   ├── src/
│   ├── tests/
│   ├── config/
│   ├── data/
│   ├── notebooks/
│   └── scripts/
│
├── proposal_4_coastal_risk/               # Proposal 4
│   ├── README.md
│   ├── src/
│   ├── data/
│   ├── config/
│   ├── docs/
│   ├── notebooks/
│   ├── results/
│   └── tests/
│
└── proposal_5_climate_sensitivity/        # Proposal 5
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── setup.py
    ├── config/
    ├── data/
    ├── src/
    ├── notebooks/
    ├── tests/
    ├── docs/
    └── results/
```

## Key Stakeholders

### African Institutions
- **IGAD**: Intergovernmental Authority on Development (Proposal 1)
- **National Meteorological Services**: All proposals
- **ACMAD**: African Centre of Meteorological Applications for Development
- **ICPAC**: IGAD Climate Prediction and Applications Centre
- **SASSCAL**: Southern African Science Service Centre for Climate Change

### Local Communities
- Farming cooperatives (Proposal 1)
- Coastal communities (Proposal 4)
- City planning departments (Proposal 4)
- Disaster management agencies (Proposals 2, 4)

### International Partners
- CMIP modeling centers (Proposal 5)
- IPCC authors (Proposals 3, 5)
- Regional climate centers
- Research institutions

## Expected Outcomes

### Scientific Outputs
- **15+ peer-reviewed publications** in high-impact journals
- **5 open-source frameworks** for climate risk assessment
- **Novel methodologies** combining physics and machine learning
- **Curated datasets** for African climate research

### Operational Systems
- **Drought early warning** operational in 8+ African countries
- **Flood prediction** deployed in 5 major West African basins
- **Coastal risk assessments** for 4 major cities
- **Teleconnection catalog** for African climate forecasting

### Capacity Building
- **20+ African scientists** trained in advanced climate methods
- **Workshops and training materials** for local institutions
- **Technology transfer** to national meteorological services
- **Sustainable operational pathways** for local ownership

### Societal Impact
- **Reduced casualties** from extreme weather events
- **Improved agricultural planning** through better drought forecasts
- **Enhanced coastal resilience** in vulnerable cities
- **Informed climate adaptation** decisions across Africa

## Timeline Summary

### Year 1 (2025-2026): Foundation
- Data pipeline development across all proposals
- Stakeholder mapping and engagement
- Baseline model establishment
- Pilot implementations

### Year 2 (2026-2027): Development
- Model refinement and validation
- Integration of teleconnections (Proposal 3)
- Expansion of operational domains
- Initial publications

### Year 3 (2027-2028): Integration
- Cross-proposal synergies activated
- Operational deployment in pilot regions
- Stakeholder feedback integration
- Capacity building intensifies

### Year 4 (2028-2029): Transition
- Full operational deployment
- Technology transfer to local institutions
- Sustainability planning
- Final documentation and publications

## Budget Summary

| Proposal | Budget | Primary Focus |
|----------|--------|---------------|
| 1. Drought Early Warning | €650,000 | East & Southern Africa drought |
| 2. Flood Prediction | €580,000 | West African river basins |
| 3. Climate Connections | €500,000 | Global teleconnections |
| 4. Coastal Risk | TBD | African coastal cities |
| 5. Climate Sensitivity | €910,000 | Global climate sensitivity |
| **Total** | **€2,640,000+** | **Africa-focused climate science** |

### Budget Allocation Across Program
- **Personnel** (58%): PhD students, postdocs, research engineers
- **Computing Infrastructure** (12%): HPC access, cloud storage, data processing
- **Capacity Building** (10%): Training, workshops, technology transfer
- **Stakeholder Engagement** (8%): Community consultations, co-development
- **Data & Fieldwork** (7%): Data acquisition, field campaigns, validation
- **Dissemination** (5%): Publications, conferences, outreach

## Technical Approach

### Common Methodological Themes

1. **Bayesian Frameworks**: All proposals use Bayesian methods for uncertainty quantification
2. **Physics-Informed ML**: Integration of physical constraints with machine learning (Proposals 1, 2, 3)
3. **Multi-Scale Modeling**: Hierarchical approaches across spatial/temporal scales
4. **Ensemble Methods**: Multiple models/methods for robust predictions
5. **Open Science**: All code and data will be publicly available

### Computational Requirements
- **Data Storage**: ~10 TB across all proposals
- **Computing**: HPC access for Bayesian MCMC, climate model analysis
- **Software Stack**: Python (primary), R, Julia for specialized analyses
- **Infrastructure**: Cloud computing for operational systems

## Getting Started

Each proposal has detailed documentation in its respective directory:

1. Navigate to the proposal directory of interest
2. Read the detailed `README.md` for that proposal
3. Review methodology in `docs/` subdirectories
4. Explore code in `src/` directories
5. See `notebooks/` for analysis examples

### For Researchers
- All methodologies are documented in proposal-specific `docs/` folders
- Source code will be released under open licenses
- Data sources and requirements are specified in each proposal

### For Stakeholders
- Each proposal includes stakeholder engagement plans
- Quarterly updates and annual workshops planned
- Co-development opportunities for operational tools

### For Collaborators
- Contact the PI for collaboration opportunities
- Proposals designed for modular participation
- Integration points across proposals

## Publications Strategy

### Target Journals
- **High Impact**: Nature Climate Change, Science, Science Advances
- **Domain Specific**: Nature Geoscience, GRL, Journal of Climate
- **Methods**: GMD, JAMES, Environmental Modelling & Software
- **Regional**: African Journal of Science, African Climate and Development Initiative

### Open Access Commitment
All publications will be open access to ensure accessibility to African researchers and institutions.

## Data and Code Availability

### Principles
- **FAIR Data**: Findable, Accessible, Interoperable, Reusable
- **Open Source**: All code released under permissive licenses (MIT/Apache 2.0)
- **Reproducibility**: Docker containers, documented workflows
- **DOI Assignment**: All datasets and code versions archived on Zenodo

### Repositories
- **GitHub**: Active development and collaboration
- **Zenodo**: Permanent archival with DOIs
- **Institutional Repositories**: African institutional access

## Contact and Collaboration

**Principal Investigator**: Moses Kolleh
**Title**: Environmental Scientist & Sustainable AI Researcher
**Institution**: Digital Society School, Amsterdam University of Applied Sciences
**GitHub**: [moseskolleh/climatematchProjects](https://github.com/moseskolleh/climatematchProjects)

### Collaboration Opportunities
- African institutions interested in technology transfer
- Climate modeling centers for validation studies
- Funding agencies supporting African climate science
- NGOs and development agencies for operational deployment

## Acknowledgments

This research program is designed to address critical gaps in African climate science while building local capacity and ensuring long-term sustainability. The proposals recognize that effective climate adaptation requires not just sophisticated methods, but also meaningful engagement with local communities and institutions.

## License

Individual proposals may have specific licenses. See each proposal directory for details.

---

## Quick Navigation

| Proposal | Focus Area | Budget | Link |
|----------|-----------|--------|------|
| **1** | Drought Early Warning | €650k | [View](./proposal-01-drought-early-warning/) |
| **2** | Flood Prediction | €580k | [View](./proposal_2_flood_prediction/) |
| **3** | Climate Teleconnections | €500k | [View](./proposal_3_climate_connections/) |
| **4** | Coastal Risk | TBD | [View](./proposal_4_coastal_risk/) |
| **5** | Climate Sensitivity | €910k | [View](./proposal_5_climate_sensitivity/) |

---

*This research program represents a comprehensive approach to advancing climate science in Africa through innovative methods, meaningful stakeholder engagement, and sustainable capacity building.*

*Last Updated: November 2025*
