# Methodology: Theory-Guided Discovery of Climate System Connections

## 1. Overview

This document details the comprehensive methodology for discovering and validating climate teleconnections using theory-guided machine learning approaches. Our framework integrates physical understanding from atmospheric dynamics with modern causal discovery methods to identify robust, interpretable climate connections.

## 2. Theoretical Foundation

### 2.1 Dynamical Systems Theory

#### Rossby Wave Dynamics
Climate teleconnections are primarily mediated by Rossby waves - large-scale atmospheric waves that transport energy and momentum. Our search is constrained by:

**Dispersion Relation**:
```
ω = -β*k / (k² + l² + f²/N²H²)
```
where:
- ω: wave frequency
- k, l: horizontal wavenumbers
- β: meridional gradient of Coriolis parameter
- f: Coriolis parameter
- N: buoyancy frequency
- H: scale height

**Implications for Pattern Search**:
- Wave patterns must have appropriate phase speeds (5-15 m/s for typical Rossby waves)
- Spatial structures constrained by wavenumber relationships
- Temporal lags consistent with wave propagation times

#### Conservation Laws

**Potential Vorticity (PV)**:
```
PV = (ζ + f) / h
```
- Conserved following fluid parcels in adiabatic, frictionless flow
- Provides constraint on pattern evolution
- Helps identify causal pathways vs. spurious correlations

**Energy Conservation**:
- Kinetic energy + potential energy + internal energy = constant
- Teleconnection patterns must satisfy energy budget closure
- Energy sources and sinks must be physically consistent

### 2.2 Known Timescales

Our analysis focuses on timescales relevant to atmospheric variability:

| Phenomenon | Timescale | Relevance |
|------------|-----------|-----------|
| Synoptic weather | 3-7 days | Excluded (too short) |
| MJO | 30-90 days | Key intraseasonal signal |
| ENSO | 2-7 years | Dominant interannual variability |
| Decadal variability | 10-30 years | Climate regime shifts |
| Climate change trend | > 30 years | Removed via detrending |

## 3. Data Processing Pipeline

### 3.1 Data Acquisition

#### Primary Reanalysis Datasets

**ERA5 (ECMWF Reanalysis v5)**:
- Temporal coverage: 1979-present
- Spatial resolution: 0.25° × 0.25°
- Temporal resolution: Hourly
- Variables: Full 3D atmospheric state

**MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)**:
- Temporal coverage: 1980-present
- Spatial resolution: 0.5° × 0.625°
- Temporal resolution: Hourly
- Variables: Emphasis on aerosols and composition

**JRA-55 (Japanese 55-year Reanalysis)**:
- Temporal coverage: 1958-present
- Spatial resolution: 1.25° × 1.25°
- Temporal resolution: 6-hourly
- Variables: Long-term consistency

#### Key Variables

1. **Geopotential Height** (Z):
   - Levels: 1000, 850, 500, 250, 100 hPa
   - Primary diagnostic for wave patterns
   - Monthly means and anomalies

2. **Sea Surface Temperature** (SST):
   - Global coverage
   - Boundary forcing for atmosphere
   - Monthly means

3. **Outgoing Longwave Radiation** (OLR):
   - Proxy for tropical convection
   - Daily values aggregated to monthly
   - Focus on 20°S-20°N

4. **Precipitation**:
   - Critical for African impacts
   - Regional focus on Sahel, East Africa, Southern Africa
   - Monthly totals

5. **Wind Fields** (u, v):
   - 850 hPa and 200 hPa levels
   - For momentum flux calculations
   - Monthly means

### 3.2 Quality Control

#### Step 1: Data Integrity Checks
```python
def quality_control(data):
    """
    Comprehensive quality control pipeline
    """
    # Check for missing values
    missing_fraction = data.isnull().sum() / len(data)
    assert missing_fraction < 0.05, "Too many missing values"

    # Check for physically unrealistic values
    assert data.within_physical_bounds(), "Unphysical values detected"

    # Check for temporal consistency
    assert data.check_temporal_continuity(), "Temporal gaps detected"

    # Check for spatial consistency
    assert data.check_spatial_consistency(), "Spatial anomalies detected"

    return data
```

#### Step 2: Gap Filling
- Use linear interpolation for gaps < 3 months
- Use climatological mean for gaps 3-6 months
- Exclude periods with gaps > 6 months from analysis

#### Step 3: Outlier Detection
- Identify values > 5σ from climatological mean
- Flag for manual inspection
- Replace with climatological value if confirmed erroneous

### 3.3 Preprocessing

#### Detrending
Remove long-term climate change signal to focus on variability:

```python
def detrend_data(data):
    """
    Remove linear trend and low-frequency variability
    """
    # Linear detrending
    data_detrended = signal.detrend(data, axis=0, type='linear')

    # High-pass filter to remove >30-year variability
    data_filtered = apply_highpass_filter(data_detrended, cutoff=30)

    return data_filtered
```

#### Deseasonalization
Remove annual cycle to focus on anomalies:

```python
def remove_seasonal_cycle(data):
    """
    Calculate and remove climatological seasonal cycle
    """
    # Calculate monthly climatology (1981-2010 baseline)
    climatology = data.sel(time=slice('1981', '2010')).groupby('time.month').mean('time')

    # Remove seasonal cycle
    anomalies = data.groupby('time.month') - climatology

    return anomalies
```

#### Spatial/Temporal Filtering

**Spatial**:
- Focus on large-scale patterns (>1000 km)
- Apply Gaussian filter with 500 km std dev
- Reduce small-scale noise

**Temporal**:
- Apply bandpass filters for specific phenomena:
  - MJO: 30-90 day bandpass
  - Interannual: 1-7 year bandpass
  - Decadal: 8-15 year bandpass

### 3.4 Domain Decomposition

Divide analysis into regional domains to reduce computational burden:

1. **Source Regions**: Known climate drivers
   - Tropical Pacific (ENSO)
   - Tropical Indian Ocean (IOD)
   - Tropical Atlantic
   - North Atlantic (NAO)
   - Arctic (AO)

2. **Target Regions**: African climate zones
   - Sahel (10°N-20°N, 20°W-40°E)
   - West Africa (5°N-15°N, 15°W-10°E)
   - East Africa (10°S-15°N, 25°E-52°E)
   - Southern Africa (35°S-15°S, 10°E-40°E)

## 4. Hypothesis-Driven Search

### 4.1 Candidate Generation

Based on dynamical theory, we generate hypotheses about potential teleconnections:

#### Theory-Based Candidates

1. **Tropical-Extratropical Connections**:
   - Hypothesis: Tropical convection anomalies excite Rossby waves
   - Mechanism: Latent heat release → upper-level divergence → Rossby wave source
   - Search: OLR anomalies (tropics) → Z500 patterns (extratropics) → African rainfall

2. **Stratosphere-Troposphere Coupling**:
   - Hypothesis: Stratospheric polar vortex variability affects tropospheric circulation
   - Mechanism: Downward wave coupling
   - Search: Z50 anomalies (polar) → Z500 patterns → African climate

3. **Ocean-Atmosphere Interactions**:
   - Hypothesis: SST gradients force atmospheric circulation changes
   - Mechanism: Diabatic heating → pressure gradients → wind response
   - Search: SST patterns → circulation indices → African rainfall

4. **Monsoon Teleconnections**:
   - Hypothesis: Asian monsoon variability affects African climate
   - Mechanism: Upper-level divergence → Rossby wave train
   - Search: Indian monsoon indices → Mediterranean climate → Sahel rainfall

### 4.2 Physical Constraints

For each candidate, we enforce constraints:

```python
class PhysicalConstraints:
    """
    Enforce physical consistency in teleconnection search
    """
    def __init__(self):
        self.rossby_wave_checker = RossbyWaveValidator()
        self.energy_budget = EnergyBudgetAnalyzer()
        self.timescale_validator = TimescaleChecker()

    def validate_candidate(self, pattern):
        """
        Check if pattern satisfies physical constraints
        """
        # Check Rossby wave dynamics
        if not self.rossby_wave_checker.is_valid(pattern):
            return False

        # Check energy conservation
        if not self.energy_budget.closes(pattern):
            return False

        # Check timescales
        if not self.timescale_validator.is_physical(pattern):
            return False

        return True
```

#### Rossby Wave Validation
```python
def validate_rossby_wave(pattern):
    """
    Check if pattern consistent with Rossby wave dynamics
    """
    # Calculate phase speed
    phase_speed = calculate_phase_speed(pattern)

    # Theoretical Rossby wave speed
    beta = 1.6e-11  # m^-1 s^-1
    L = pattern.wavelength
    theoretical_speed = -beta * L**2 / (4 * np.pi**2)

    # Allow 50% tolerance
    return abs(phase_speed - theoretical_speed) / theoretical_speed < 0.5
```

## 5. Causal Discovery Methods

### 5.1 Granger Causality

#### Methodology
Test if past values of variable X improve prediction of variable Y beyond Y's own history.

**Vector Autoregression (VAR) Model**:
```
Y_t = Σ(α_i * Y_{t-i}) + Σ(β_j * X_{t-j}) + ε_t
```

**Granger Causality Test**:
- Null hypothesis: β_j = 0 for all j (X does not Granger-cause Y)
- F-test for significance of β coefficients
- Reject null if p < 0.01 (after FDR correction)

#### Implementation
```python
def granger_causality_test(X, Y, max_lag=12):
    """
    Test if X Granger-causes Y
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Combine into dataframe
    data = pd.DataFrame({'Y': Y, 'X': X})

    # Run test for lags 1 to max_lag
    results = grangercausalitytests(data, max_lag, verbose=False)

    # Extract p-values
    p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]

    # Find optimal lag (minimum p-value)
    optimal_lag = np.argmin(p_values) + 1
    p_value = min(p_values)

    return {
        'granger_causes': p_value < 0.01,
        'p_value': p_value,
        'optimal_lag': optimal_lag
    }
```

### 5.2 Convergent Cross Mapping (CCM)

For nonlinear dynamical systems, CCM detects causality through state space reconstruction.

#### Methodology
1. Reconstruct attractor for Y using time-delay embedding
2. For each point in Y's attractor, find nearest neighbors
3. Use those neighbors to predict X
4. If prediction skill increases with library size, X causes Y

#### Implementation
```python
def convergent_cross_mapping(X, Y, E=3, tau=1, lib_sizes=None):
    """
    Test causality using CCM

    Args:
        X, Y: Time series
        E: Embedding dimension
        tau: Time delay
        lib_sizes: Library sizes to test
    """
    if lib_sizes is None:
        lib_sizes = np.arange(50, len(X), 50)

    skills = []

    for L in lib_sizes:
        # Reconstruct Y's attractor
        attractor_Y = embed_timeseries(Y, E, tau)

        # Sample library of size L
        lib_indices = np.random.choice(len(attractor_Y), L, replace=False)

        # For each point, predict X using Y's attractor
        predictions = []
        actuals = []

        for i in range(len(attractor_Y)):
            if i in lib_indices:
                continue

            # Find E+1 nearest neighbors in Y's attractor
            neighbors = find_nearest_neighbors(
                attractor_Y[i],
                attractor_Y[lib_indices],
                k=E+1
            )

            # Predict X using weighted average
            weights = exponential_weights(neighbors)
            prediction = np.sum(weights * X[lib_indices[neighbors]])

            predictions.append(prediction)
            actuals.append(X[i])

        # Calculate prediction skill
        skill = np.corrcoef(predictions, actuals)[0, 1]
        skills.append(skill)

    # Test for convergence (increasing skill with library size)
    slope, _, r_value, p_value, _ = scipy.stats.linregress(lib_sizes, skills)

    return {
        'converges': (slope > 0) and (p_value < 0.01),
        'p_value': p_value,
        'skills': skills,
        'lib_sizes': lib_sizes
    }
```

### 5.3 Transfer Entropy

Quantifies directed information flow from X to Y.

#### Methodology
```
TE(X→Y) = Σ p(y_t, y_{t-1}, x_{t-1}) * log[p(y_t|y_{t-1}, x_{t-1}) / p(y_t|y_{t-1})]
```

#### Implementation
```python
def transfer_entropy(X, Y, k=1, l=1):
    """
    Calculate transfer entropy from X to Y

    Args:
        X, Y: Time series
        k: History length for Y
        l: History length for X
    """
    # Discretize time series for probability estimation
    X_discrete = discretize_timeseries(X, bins=10)
    Y_discrete = discretize_timeseries(Y, bins=10)

    # Build joint and conditional probability distributions
    # p(y_t, y_{t-1:t-k}, x_{t-1:t-l})
    joint_prob = estimate_joint_probability(
        Y_discrete[k:],
        Y_discrete[:k],
        X_discrete[:l]
    )

    # p(y_t | y_{t-1:t-k}, x_{t-1:t-l})
    cond_prob_with_X = conditional_probability(joint_prob, condition_on=[1,2])

    # p(y_t | y_{t-1:t-k})
    cond_prob_without_X = conditional_probability(joint_prob, condition_on=[1])

    # Calculate transfer entropy
    TE = np.sum(
        joint_prob * np.log2(cond_prob_with_X / cond_prob_without_X)
    )

    # Statistical significance via permutation test
    TE_null = []
    for _ in range(1000):
        X_shuffled = np.random.permutation(X)
        TE_null.append(transfer_entropy_calculation(X_shuffled, Y, k, l))

    p_value = np.mean(TE_null >= TE)

    return {
        'transfer_entropy': TE,
        'p_value': p_value,
        'significant': p_value < 0.01
    }
```

### 5.4 Structural Causal Models

Use Pearl's causal framework with directed acyclic graphs (DAGs).

#### Methodology
1. Define causal graph structure based on physical understanding
2. Estimate structural equation parameters from data
3. Test causal hypotheses using do-calculus
4. Validate with counterfactual predictions

#### Implementation
```python
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

def structural_causal_model(data, graph_structure):
    """
    Fit and test structural causal model

    Args:
        data: DataFrame with variables
        graph_structure: Dict defining DAG edges
    """
    # Create causal graph
    causal_graph = CausalGraphicalModel(
        nodes=data.columns,
        edges=graph_structure
    )

    # Estimate structural equations
    equations = {}
    for node in causal_graph.dag.nodes():
        parents = list(causal_graph.dag.predecessors(node))

        if len(parents) == 0:
            # Exogenous variable
            equations[node] = lambda: data[node]
        else:
            # Regress on parents
            X = data[parents]
            y = data[node]
            model = LinearRegression().fit(X, y)
            equations[node] = lambda X: model.predict(X)

    # Perform do-calculus interventions
    def do_intervention(variable, value):
        """
        Simulate intervention: set variable to value
        """
        # Modify graph to remove incoming edges to variable
        modified_graph = causal_graph.copy()
        modified_graph.dag.remove_edges_from(
            [(p, variable) for p in modified_graph.dag.predecessors(variable)]
        )

        # Simulate from modified graph
        simulated_data = simulate_from_graph(modified_graph, equations, variable, value)

        return simulated_data

    return {
        'graph': causal_graph,
        'equations': equations,
        'do_intervention': do_intervention
    }
```

## 6. Statistical Validation Framework

### 6.1 Bootstrap Confidence Intervals

Generate confidence intervals for all discovered relationships:

```python
def bootstrap_confidence_interval(X, Y, statistic_func, n_bootstrap=10000, alpha=0.01):
    """
    Calculate bootstrap confidence interval

    Args:
        X, Y: Time series
        statistic_func: Function to calculate statistic (e.g., correlation)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
    """
    observed_statistic = statistic_func(X, Y)

    bootstrap_statistics = []
    n = len(X)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices]
        Y_boot = Y[indices]

        # Calculate statistic
        boot_stat = statistic_func(X_boot, Y_boot)
        bootstrap_statistics.append(boot_stat)

    # Calculate percentile confidence interval
    lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
    upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)

    return {
        'observed': observed_statistic,
        'ci_lower': lower,
        'ci_upper': upper,
        'bootstrap_distribution': bootstrap_statistics
    }
```

### 6.2 False Discovery Rate Control

Control for multiple comparisons using Benjamini-Hochberg procedure:

```python
def fdr_correction(p_values, alpha=0.01):
    """
    Apply Benjamini-Hochberg FDR correction

    Args:
        p_values: Array of p-values
        alpha: FDR level
    """
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate critical values
    m = len(p_values)
    critical_values = (np.arange(1, m+1) / m) * alpha

    # Find largest i where p_i <= (i/m)*alpha
    significant = sorted_p <= critical_values

    if np.any(significant):
        threshold_index = np.where(significant)[0][-1]
        threshold = sorted_p[threshold_index]
    else:
        threshold = 0

    # Mark discoveries
    discoveries = p_values <= threshold

    return {
        'discoveries': discoveries,
        'threshold': threshold,
        'n_discoveries': np.sum(discoveries),
        'fdr': alpha
    }
```

### 6.3 Cross-Validation Across Reanalysis Products

Test if discoveries are robust across datasets:

```python
def cross_reanalysis_validation(pattern, datasets):
    """
    Validate pattern across multiple reanalysis products

    Args:
        pattern: Teleconnection pattern to validate
        datasets: List of reanalysis datasets ['ERA5', 'MERRA2', 'JRA55']
    """
    correlations = []
    p_values = []

    # Reference dataset (ERA5)
    reference = datasets['ERA5']
    reference_pattern = extract_pattern(reference, pattern)

    # Test against other datasets
    for name, dataset in datasets.items():
        if name == 'ERA5':
            continue

        # Extract same pattern from this dataset
        test_pattern = extract_pattern(dataset, pattern)

        # Calculate pattern correlation
        corr, p_val = scipy.stats.pearsonr(reference_pattern, test_pattern)
        correlations.append(corr)
        p_values.append(p_val)

    # Pattern is validated if correlation > 0.7 with all datasets
    validated = all(np.array(correlations) > 0.7) and all(np.array(p_values) < 0.01)

    return {
        'validated': validated,
        'correlations': dict(zip([d for d in datasets.keys() if d != 'ERA5'], correlations)),
        'p_values': dict(zip([d for d in datasets.keys() if d != 'ERA5'], p_values)),
        'mean_correlation': np.mean(correlations)
    }
```

### 6.4 Temporal Stability Testing

Check if relationships hold across different time periods:

```python
def temporal_stability_test(X, Y, n_periods=4):
    """
    Test temporal stability by splitting into sub-periods

    Args:
        X, Y: Time series
        n_periods: Number of sub-periods
    """
    n = len(X)
    period_length = n // n_periods

    correlations = []

    for i in range(n_periods):
        start = i * period_length
        end = (i + 1) * period_length

        X_period = X[start:end]
        Y_period = Y[start:end]

        corr, _ = scipy.stats.pearsonr(X_period, Y_period)
        correlations.append(corr)

    # Test for consistency: all periods have same sign and magnitude
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    same_sign = all(np.sign(correlations) == np.sign(mean_corr))

    # Relationship is stable if CV < 0.5 and all same sign
    coefficient_of_variation = std_corr / abs(mean_corr)
    stable = (coefficient_of_variation < 0.5) and same_sign

    return {
        'stable': stable,
        'correlations': correlations,
        'mean': mean_corr,
        'std': std_corr,
        'cv': coefficient_of_variation
    }
```

## 7. Physical Interpretation Protocol

### 7.1 Composite Analysis

Create composites of high/low states to visualize patterns:

```python
def composite_analysis(index, fields, threshold=1.0):
    """
    Create composite maps for high/low index states

    Args:
        index: Climate index time series (standardized)
        fields: Dictionary of climate fields
        threshold: Standard deviation threshold
    """
    # Identify high and low states
    high_states = index > threshold
    low_states = index < -threshold

    composites = {}

    for field_name, field_data in fields.items():
        # Calculate composites
        high_composite = field_data[high_states].mean(axis=0)
        low_composite = field_data[low_states].mean(axis=0)
        difference = high_composite - low_composite

        # Statistical significance (t-test)
        t_stat, p_val = scipy.stats.ttest_ind(
            field_data[high_states],
            field_data[low_states],
            axis=0
        )

        composites[field_name] = {
            'high': high_composite,
            'low': low_composite,
            'difference': difference,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    return composites
```

### 7.2 Energy and Momentum Budget Analysis

Diagnose physical mechanisms through budget calculations:

```python
def energy_budget_analysis(fields):
    """
    Calculate atmospheric energy budget

    Components:
    - Kinetic energy tendency
    - Potential energy tendency
    - Energy conversion terms
    - Boundary fluxes
    """
    u = fields['u_wind']  # zonal wind
    v = fields['v_wind']  # meridional wind
    T = fields['temperature']
    omega = fields['omega']  # vertical velocity

    # Kinetic energy
    KE = 0.5 * (u**2 + v**2)

    # Available potential energy
    T_mean = T.mean(dim=['lat', 'lon'])
    APE = (cp / (2 * T_mean)) * (T - T_mean)**2

    # Energy conversion (baroclinic)
    conversion = (R / p) * omega * (T - T_mean)

    # Calculate tendencies
    dKE_dt = calculate_tendency(KE)
    dAPE_dt = calculate_tendency(APE)

    # Budget closure
    residual = dKE_dt + dAPE_dt - conversion

    return {
        'kinetic_energy': KE,
        'potential_energy': APE,
        'conversion': conversion,
        'ke_tendency': dKE_dt,
        'ape_tendency': dAPE_dt,
        'residual': residual,
        'closure': (abs(residual) / (abs(dKE_dt) + abs(dAPE_dt))) < 0.1
    }
```

### 7.3 Climate Model Experiments

Validate mechanisms using climate model perturbation experiments:

```python
def design_model_experiments(teleconnection):
    """
    Design climate model experiments to test teleconnection mechanism

    Args:
        teleconnection: Discovered teleconnection object
    """
    experiments = []

    # Experiment 1: Idealized forcing at source region
    experiments.append({
        'name': 'idealized_forcing',
        'description': 'Apply idealized heating/cooling at source region',
        'forcing': {
            'type': 'heating',
            'location': teleconnection.source_region,
            'amplitude': '1 K/day',
            'duration': '30 days'
        },
        'analysis': 'Track wave propagation to target region'
    })

    # Experiment 2: Remove specific physical process
    experiments.append({
        'name': 'mechanism_blocking',
        'description': 'Remove hypothesized mechanism',
        'modification': {
            'disable': teleconnection.proposed_mechanism,
        },
        'analysis': 'Check if teleconnection disappears'
    })

    # Experiment 3: Sensitivity to background state
    experiments.append({
        'name': 'state_dependence',
        'description': 'Test under different climate states',
        'scenarios': ['warm_SST', 'cold_SST', 'strong_jet', 'weak_jet'],
        'analysis': 'Assess non-stationarity'
    })

    return experiments
```

### 7.4 Expert Evaluation Framework

Structured protocol for expert review:

```python
def expert_evaluation_protocol(teleconnection):
    """
    Generate evaluation questionnaire for dynamical meteorologists
    """
    questionnaire = {
        'pattern_description': teleconnection.describe(),
        'questions': [
            {
                'q': 'Is the spatial pattern physically plausible?',
                'options': ['Yes', 'Partially', 'No'],
                'follow_up': 'Please explain your reasoning'
            },
            {
                'q': 'Are the timescales consistent with proposed mechanism?',
                'options': ['Yes', 'Uncertain', 'No'],
                'follow_up': 'What timescales would you expect?'
            },
            {
                'q': 'Does the pattern respect relevant conservation laws?',
                'options': ['Yes', 'Uncertain', 'No'],
                'follow_up': 'Which conservation laws are relevant?'
            },
            {
                'q': 'Can you propose alternative mechanisms?',
                'options': ['Yes', 'No'],
                'follow_up': 'Please describe alternative mechanisms'
            },
            {
                'q': 'Overall confidence in this teleconnection:',
                'options': ['High (>80%)', 'Medium (50-80%)', 'Low (<50%)'],
                'follow_up': 'What evidence would increase your confidence?'
            }
        ]
    }

    return questionnaire
```

## 8. Integration with Other Proposals

### 8.1 Enhancing Drought Prediction (Proposal 1)

Use discovered teleconnections as predictors:

```python
def integrate_with_drought_prediction(teleconnections, drought_model):
    """
    Add teleconnection indices as features to drought prediction model
    """
    # Extract teleconnection indices
    for tc in teleconnections:
        if tc.affects_region('Sahel') and tc.lead_time >= 30:  # 1+ month lead
            # Add as predictor
            drought_model.add_feature(
                name=tc.name,
                data=tc.index,
                lead_time=tc.lead_time,
                uncertainty=tc.uncertainty
            )

    # Retrain model with new features
    drought_model.train()

    # Assess improvement
    improvement = drought_model.evaluate() - drought_model.baseline_skill

    return improvement
```

### 8.2 Enhancing Flood Prediction (Proposal 2)

Incorporate large-scale drivers:

```python
def integrate_with_flood_prediction(teleconnections, flood_model):
    """
    Use teleconnections to improve precipitation forecasts for flood model
    """
    # Identify teleconnections affecting precipitation
    precip_tc = [tc for tc in teleconnections if tc.target_variable == 'precipitation']

    # Create ensemble of precipitation scenarios
    scenarios = []
    for tc in precip_tc:
        scenario = flood_model.generate_scenario(
            driver=tc.index,
            relationship=tc.regression_model
        )
        scenarios.append(scenario)

    # Run flood model for each scenario
    flood_ensemble = flood_model.run_ensemble(scenarios)

    return flood_ensemble
```

## 9. Quality Assurance Checklist

Before publishing any discovered teleconnection, verify:

### Statistical Criteria
- [ ] p-value < 0.01 after FDR correction
- [ ] Effect size (Cohen's d) > 0.5
- [ ] Bootstrap CI excludes zero
- [ ] Validated across ≥2 reanalysis products
- [ ] Temporally stable across ≥2 independent periods

### Physical Criteria
- [ ] Consistent with Rossby wave dynamics
- [ ] Satisfies energy budget closure (residual < 10%)
- [ ] Timescales physically reasonable
- [ ] Spatial pattern coherent and interpretable
- [ ] Mechanism supported by composite analysis

### Model Validation
- [ ] Reproduced in ≥5 CMIP6 models
- [ ] Mechanism confirmed via idealized experiments
- [ ] Sensitivity to background state understood

### Expert Review
- [ ] Evaluated by ≥3 independent experts
- [ ] Majority confidence ≥ "Medium"
- [ ] Alternative mechanisms considered and tested

## 10. Documentation Standards

All discovered teleconnections must be documented with:

1. **Summary Card**:
   - Name and brief description
   - Source and target regions
   - Typical timescale and lead time
   - Magnitude of impact
   - Confidence level

2. **Detailed Technical Report**:
   - Full statistical analysis
   - Physical mechanism description
   - Validation results
   - Limitations and uncertainties

3. **Visualization Package**:
   - Composite maps
   - Time series of index
   - Scatterplots with regression
   - Budget diagnostics

4. **Code Repository**:
   - Reproducible analysis scripts
   - Data processing pipeline
   - Validation tests

---

*This methodology provides a comprehensive, rigorous framework for discovering and validating climate teleconnections. Adherence to these protocols ensures that our results are scientifically sound, physically interpretable, and operationally useful.*
