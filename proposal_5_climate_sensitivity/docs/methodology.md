# Methodology: Multi-Constraint Framework for Climate Sensitivity

## Table of Contents
1. [Introduction](#introduction)
2. [Constraint Types](#constraint-types)
3. [Integration Framework](#integration-framework)
4. [Uncertainty Quantification](#uncertainty-quantification)
5. [Validation Approach](#validation-approach)

## Introduction

Climate sensitivity, typically expressed as Equilibrium Climate Sensitivity (ECS), quantifies the long-term global mean surface temperature change resulting from a doubling of atmospheric CO₂ concentration. Accurately constraining ECS is crucial for:

- Projecting future climate change
- Assessing climate risks
- Informing mitigation policy
- Setting carbon budgets

This methodology combines multiple independent lines of evidence (constraints) to narrow the uncertainty range of ECS while maintaining scientific conservatism and transparency.

## Constraint Types

### 1. Paleoclimate Constraints

Paleoclimate periods provide natural experiments where both forcing changes and temperature responses are observable.

#### 1.1 Last Glacial Maximum (LGM)

**Period**: ~21,000 years ago

**Approach**:
```python
# Simplified constraint equation
ΔT_LGM = observed temperature anomaly (~-6°C globally)
ΔF_LGM = radiative forcing change from:
    - CO₂ reduction (190 vs 280 ppm)
    - Ice sheet albedo
    - Vegetation changes
    - Dust aerosols

ECS_LGM = ΔT_LGM / (ΔF_LGM / ΔF_2×CO₂)
```

**Data Sources**:
- Temperature proxies: Ice cores, marine sediments, pollen, noble gases
- Forcing components: Ice sheet reconstructions, vegetation models, dust deposition
- Model simulations: PMIP3/PMIP4 LGM experiments

**Uncertainties**:
- Proxy temperature interpretation: ±1-2°C
- Global mean calculation from sparse proxies: ±0.5-1°C
- Forcing estimates (especially aerosols): ±30%
- State-dependence of feedbacks: Unknown

**Implementation Steps**:

1. **Compile proxy temperature database**
   ```python
   def compile_lgm_temperatures():
       """
       Aggregate LGM temperature proxies from multiple sources
       """
       sources = ['ice_cores', 'marine_sediments', 'pollen', 'noble_gases']
       proxies = []

       for source in sources:
           data = load_proxy_data(source)
           # Apply calibration uncertainties
           calibrated = apply_calibration(data)
           proxies.append(calibrated)

       return combine_proxies(proxies)
   ```

2. **Calculate global mean anomaly**
   ```python
   def calculate_lgm_global_mean(proxy_data):
       """
       Spatially interpolate and calculate global mean with uncertainties
       """
       # Use data assimilation approach
       prior = load_model_ensemble('PMIP4')
       posterior = data_assimilation(proxy_data, prior)

       # Bootstrap for uncertainty
       n_bootstrap = 10000
       global_means = []
       for i in range(n_bootstrap):
           resampled = bootstrap_proxies(proxy_data)
           gm = calculate_global_mean(resampled, posterior)
           global_means.append(gm)

       return np.percentile(global_means, [5, 50, 95])
   ```

3. **Estimate forcing components**
   ```python
   def calculate_lgm_forcing():
       """
       Calculate total radiative forcing change at LGM
       """
       # CO₂ forcing
       F_CO2 = 5.35 * np.log(190/280)  # W/m²

       # Ice sheet albedo (from reconstructions)
       F_ice = calculate_ice_sheet_forcing()  # ~-3.5 W/m²

       # Vegetation albedo
       F_veg = calculate_vegetation_forcing()  # ~-1.0 W/m²

       # Dust
       F_dust = calculate_dust_forcing()  # ~-1.0 W/m²

       # Total with uncertainties
       F_total = propagate_uncertainties(F_CO2, F_ice, F_veg, F_dust)

       return F_total
   ```

4. **Derive ECS constraint**
   ```python
   def lgm_ecs_constraint(T_lgm, F_lgm, F_2xCO2=3.7):
       """
       Calculate ECS from LGM constraint
       """
       # Account for state-dependence
       state_dependence_factor = estimate_state_dependence()

       # Calculate sensitivity
       ECS = (T_lgm / (F_lgm / F_2xCO2)) * state_dependence_factor

       return ECS
   ```

#### 1.2 Mid-Pliocene Warm Period (mPWP)

**Period**: ~3.3-3.0 million years ago

**Approach**:
- Warmer than pre-industrial (~3°C globally)
- CO₂ levels similar to present (~400 ppm)
- Provides constraint on Earth System Sensitivity (includes slow feedbacks)

**Key Advantage**: Tests climate response under warmth, complementing LGM cold constraint

**Implementation**:
```python
def mpwp_constraint():
    """
    Calculate ECS constraint from Mid-Pliocene
    """
    # Temperature reconstruction
    T_mpwp = load_pliovar_data()  # PlioVAR project data

    # CO₂ forcing
    F_CO2 = 5.35 * np.log(400/280)

    # Slower feedbacks (ice sheets, vegetation)
    F_slow = estimate_slow_feedbacks_mpwp()

    # Earth System Sensitivity
    ESS = T_mpwp / (F_CO2 / 3.7)

    # Estimate ECS (removing slow feedbacks)
    ECS = ESS - slow_feedback_contribution()

    return ECS
```

#### 1.3 Last Interglacial (LIG)

**Period**: ~129-116 thousand years ago

**Approach**:
- Peak temperatures ~1-2°C warmer than pre-industrial
- Orbital forcing different from present
- Tests tropical feedbacks under warmth

**Implementation**:
```python
def lig_constraint():
    """
    Calculate ECS constraint from Last Interglacial
    """
    # Temperature anomaly
    T_lig = compile_lig_proxies()  # ~1.5°C

    # Orbital forcing changes
    F_orbital = calculate_orbital_forcing()

    # Feedback analysis from models
    feedback_pattern = analyze_lig_feedbacks()

    # Derive ECS using feedback decomposition
    ECS = derive_ecs_from_feedbacks(T_lig, F_orbital, feedback_pattern)

    return ECS
```

### 2. Observational Constraints

Historical observations provide direct constraints on climate response to anthropogenic forcing.

#### 2.1 Energy Budget Constraint

**Approach**:
The energy budget method uses observed warming and radiative forcing to constrain ECS:

```
ΔT_obs = (ECS / F_2×CO₂) × (ΔF - ΔN)
```

Where:
- ΔT_obs = observed warming (e.g., 1850-2020)
- ΔF = effective radiative forcing change
- ΔN = top-of-atmosphere energy imbalance (ocean heat uptake)

**Implementation**:

```python
def energy_budget_constraint():
    """
    Calculate ECS from historical energy budget
    """
    # Observed temperature change
    temp_datasets = ['HadCRUT5', 'GISTEMP', 'NOAAGlobalTemp', 'Berkeley']
    T_obs = load_and_blend_temperatures(temp_datasets)
    delta_T = T_obs['2010-2020'].mean() - T_obs['1850-1900'].mean()

    # Effective radiative forcing
    F_CO2 = calculate_co2_forcing()
    F_other_ghg = calculate_other_ghg_forcing()
    F_aerosol = calculate_aerosol_forcing()  # Large uncertainty
    F_other = calculate_other_forcing()  # Land use, solar, volcanic

    delta_F = F_CO2 + F_other_ghg + F_aerosol + F_other

    # Energy imbalance (ocean heat uptake)
    delta_N = calculate_energy_imbalance()  # From CERES, Argo

    # Calculate ECS
    F_2xCO2 = 3.7  # W/m²
    ECS = (delta_T / (delta_F - delta_N)) * F_2xCO2

    # Account for pattern effects
    pattern_correction = calculate_pattern_effect()
    ECS_corrected = ECS * pattern_correction

    return ECS_corrected
```

**Key Uncertainties**:
1. **Aerosol forcing**: -1.5 to -0.5 W/m² (IPCC AR6)
2. **Ocean heat uptake**: Incomplete pre-Argo coverage
3. **Pattern effects**: Warming pattern influences effective sensitivity
4. **Internal variability**: AMO, PDO, ENSO

**Handling Aerosol Uncertainty**:
```python
def propagate_aerosol_uncertainty():
    """
    Monte Carlo sampling of aerosol forcing uncertainty
    """
    n_samples = 100000

    # Aerosol forcing distribution (from IPCC AR6)
    F_aerosol_samples = np.random.normal(-1.0, 0.25, n_samples)

    ECS_samples = []
    for F_aer in F_aerosol_samples:
        delta_F = F_CO2 + F_other_ghg + F_aer + F_other
        ECS = (delta_T / (delta_F - delta_N)) * F_2xCO2
        ECS_samples.append(ECS)

    return np.array(ECS_samples)
```

#### 2.2 Pattern-Based Constraints

Spatial patterns of warming provide additional constraints:

```python
def pattern_scaling_constraint():
    """
    Use warming patterns to constrain feedbacks
    """
    # Observed patterns
    warming_pattern = calculate_regional_warming()

    # Model patterns
    model_patterns = load_cmip6_patterns()

    # Emergent relationship between pattern and ECS
    pattern_metric = calculate_pattern_metric(warming_pattern)

    # Constrain ECS based on pattern-ECS relationship
    ECS_dist = emergent_constraint(pattern_metric, model_patterns)

    return ECS_dist
```

### 3. Process-Based Constraints

Physical understanding of individual feedbacks constrains ECS.

#### 3.1 Cloud Feedback

Clouds contribute the largest uncertainty to ECS estimates. We use multiple approaches:

**a) Cloud-Controlling Factor (CCF) Analysis**:
```python
def cloud_feedback_ccf():
    """
    Constrain cloud feedback using cloud-controlling factors
    """
    # Identify meteorological factors controlling clouds
    ccf = ['SST', 'EIS', 'omega_500', 'RH_700']

    # Observe present-day cloud-CCF relationships
    obs_relationships = analyze_satellite_clouds(ccf)

    # Evaluate models on present-day relationships
    model_skill = evaluate_models_on_ccf(obs_relationships)

    # Weight models by skill
    weighted_feedback = weight_cloud_feedback_by_skill(model_skill)

    return weighted_feedback
```

**b) Emergent Constraints**:
```python
def cloud_feedback_emergent_constraint():
    """
    Use emergent constraints on cloud feedback
    """
    # Example: Tropical low cloud feedback
    # Related to present-day cloud amount sensitivity to SST

    # Calculate predictor from observations
    predictor_obs = calculate_cloud_sst_sensitivity()

    # Model ensemble
    models = load_cmip6_models()
    predictor_models = [calculate_cloud_sst_sensitivity(m) for m in models]
    feedback_models = [get_cloud_feedback(m) for m in models]

    # Fit relationship
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(np.array(predictor_models).reshape(-1, 1), feedback_models)

    # Constrain using observations
    feedback_constrained = lr.predict(predictor_obs.reshape(-1, 1))

    # Quantify uncertainty
    residuals = feedback_models - lr.predict(np.array(predictor_models).reshape(-1, 1))
    feedback_uncertainty = np.std(residuals)

    return feedback_constrained, feedback_uncertainty
```

#### 3.2 Water Vapor and Lapse Rate Feedback

These feedbacks are better constrained than clouds:

```python
def water_vapor_lapse_rate_constraint():
    """
    Constrain WV+LR feedback from observations
    """
    # Observed vertical temperature and humidity structure
    obs_profiles = load_radiosonde_data()

    # Expected change under Clausius-Clapeyron
    theoretical_wv_feedback = calculate_cc_scaling()

    # Lapse rate modification
    observed_lapse_rate = calculate_lapse_rate_trends(obs_profiles)

    # Combined feedback
    wv_lr_feedback = theoretical_wv_feedback + observed_lapse_rate

    # Much smaller uncertainty than clouds (~0.1 W/m²/K)

    return wv_lr_feedback
```

#### 3.3 Albedo Feedback

Surface albedo feedback from ice and snow:

```python
def albedo_feedback_constraint():
    """
    Constrain albedo feedback from observations
    """
    # Sea ice trends
    sea_ice_trends = load_sea_ice_data()
    sea_ice_feedback = calculate_sea_ice_feedback(sea_ice_trends)

    # Snow cover trends
    snow_trends = load_snow_cover_data()
    snow_feedback = calculate_snow_feedback(snow_trends)

    # Combine
    albedo_feedback = sea_ice_feedback + snow_feedback

    return albedo_feedback
```

## Integration Framework

### Bayesian Multi-Constraint Integration

Constraints are combined using Bayes' theorem:

```python
def bayesian_integration(constraints, prior='uniform'):
    """
    Combine multiple constraints using Bayesian framework

    Parameters:
    -----------
    constraints : list of dict
        Each dict contains 'likelihood' function and 'name'
    prior : str or callable
        Prior distribution on ECS

    Returns:
    --------
    posterior : array
        Posterior distribution on ECS
    """
    # Define ECS range
    ecs_values = np.linspace(1.0, 6.0, 1000)

    # Prior
    if prior == 'uniform':
        prior_pdf = np.ones_like(ecs_values) / len(ecs_values)
    elif prior == 'informed':
        # Use expert elicitation or previous assessment
        prior_pdf = informed_prior(ecs_values)
    else:
        prior_pdf = prior(ecs_values)

    # Combine likelihoods
    combined_likelihood = np.ones_like(ecs_values)

    for constraint in constraints:
        # Check independence
        if not is_independent(constraint, constraints):
            print(f"Warning: {constraint['name']} may not be independent")

        # Multiply likelihoods
        likelihood = constraint['likelihood'](ecs_values)
        combined_likelihood *= likelihood

    # Posterior
    posterior = prior_pdf * combined_likelihood
    posterior /= np.trapz(posterior, ecs_values)  # Normalize

    return ecs_values, posterior
```

### Independence Testing

Critical to avoid double-counting:

```python
def test_constraint_independence(constraint_a, constraint_b, models):
    """
    Test if two constraints are independent in model space

    Uses multiple approaches:
    1. Correlation analysis
    2. Mutual information
    3. Conditional independence tests
    """
    # Calculate predictor values for both constraints across models
    predictor_a = [constraint_a.calculate_predictor(m) for m in models]
    predictor_b = [constraint_b.calculate_predictor(m) for m in models]

    # Test 1: Correlation
    correlation = np.corrcoef(predictor_a, predictor_b)[0, 1]

    # Test 2: Mutual information
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(
        np.array(predictor_a).reshape(-1, 1),
        predictor_b
    )[0]

    # Test 3: Partial correlation (controlling for ECS)
    ecs_values = [get_model_ecs(m) for m in models]
    partial_corr = calculate_partial_correlation(
        predictor_a, predictor_b, ecs_values
    )

    # Independence metrics
    independence_score = {
        'correlation': correlation,
        'mutual_information': mi,
        'partial_correlation': partial_corr,
        'independent': (abs(correlation) < 0.3 and mi < 0.1)
    }

    return independence_score
```

### Constraint Weighting

Weight constraints by reliability:

```python
def weight_constraints(constraints, validation_results):
    """
    Weight constraints based on validation performance
    """
    weights = []

    for constraint in constraints:
        # Performance in perfect model tests
        perfect_model_skill = validation_results[constraint.name]['skill']

        # Robustness across model generations
        cross_gen_stability = validation_results[constraint.name]['stability']

        # Physical plausibility
        physics_score = expert_evaluation(constraint)

        # Combined weight
        weight = (perfect_model_skill * 0.5 +
                 cross_gen_stability * 0.3 +
                 physics_score * 0.2)

        weights.append(weight)

    # Normalize
    weights = np.array(weights) / np.sum(weights)

    return weights
```

## Uncertainty Quantification

### Decomposition of Uncertainty

Separate aleatory (irreducible) and epistemic (reducible) uncertainty:

```python
def decompose_uncertainty(ecs_distribution):
    """
    Decompose ECS uncertainty into components
    """
    components = {}

    # 1. Internal variability (aleatory)
    components['internal_variability'] = estimate_internal_variability()

    # 2. Model structural uncertainty (epistemic)
    components['model_structure'] = estimate_model_structure_uncertainty()

    # 3. Forcing uncertainty (epistemic)
    components['forcing'] = estimate_forcing_uncertainty()

    # 4. Proxy/observation uncertainty (mixed)
    components['data'] = estimate_data_uncertainty()

    # 5. Constraint method uncertainty (epistemic)
    components['method'] = estimate_method_uncertainty()

    # Variance decomposition
    total_variance = np.var(ecs_distribution)

    for key in components:
        components[key + '_fraction'] = components[key] / total_variance

    return components
```

### State-Dependence Uncertainty

ECS may vary with climate state:

```python
def assess_state_dependence():
    """
    Quantify potential state-dependence of ECS
    """
    # Compare paleoclimate periods
    ecs_lgm = lgm_constraint()  # Cold state
    ecs_mpwp = mpwp_constraint()  # Warm state
    ecs_historical = historical_constraint()  # Current state

    # Test for significant differences
    state_dependence = {
        'lgm_vs_historical': statistical_test(ecs_lgm, ecs_historical),
        'mpwp_vs_historical': statistical_test(ecs_mpwp, ecs_historical),
        'range': (min(ecs_lgm, ecs_mpwp, ecs_historical),
                 max(ecs_lgm, ecs_mpwp, ecs_historical))
    }

    return state_dependence
```

## Validation Approach

### Perfect Model Experiments

Test constraint methods in controlled setting:

```python
def perfect_model_test(constraint_method, models):
    """
    Validate constraint using perfect model framework

    For each model:
    1. Treat as "truth"
    2. Use remaining models to develop constraint
    3. Apply constraint to held-out model
    4. Compare constrained estimate to true model ECS
    """
    results = []

    for test_model in models:
        # Training set (all models except test model)
        train_models = [m for m in models if m != test_model]

        # Develop constraint on training set
        constraint = constraint_method.train(train_models)

        # True ECS of test model
        true_ecs = get_model_ecs(test_model)

        # Apply constraint
        constrained_ecs_dist = constraint.apply(test_model)

        # Evaluate
        constrained_median = np.median(constrained_ecs_dist)
        constrained_90ci = np.percentile(constrained_ecs_dist, [5, 95])

        # Metrics
        bias = constrained_median - true_ecs
        coverage = (true_ecs >= constrained_90ci[0] and
                   true_ecs <= constrained_90ci[1])
        sharpness = constrained_90ci[1] - constrained_90ci[0]

        results.append({
            'model': test_model.name,
            'true_ecs': true_ecs,
            'constrained_median': constrained_median,
            'bias': bias,
            'coverage': coverage,
            'sharpness': sharpness
        })

    # Aggregate performance
    performance = {
        'mean_bias': np.mean([r['bias'] for r in results]),
        'coverage_rate': np.mean([r['coverage'] for r in results]),
        'mean_sharpness': np.mean([r['sharpness'] for r in results])
    }

    return results, performance
```

### Cross-Generation Validation

Test stability across CMIP generations:

```python
def cross_generation_validation(constraint_method):
    """
    Test constraint consistency across model generations
    """
    # Train on CMIP5
    cmip5_models = load_cmip5_models()
    constraint_cmip5 = constraint_method.train(cmip5_models)
    ecs_dist_cmip5 = constraint_cmip5.estimate_ecs()

    # Apply to CMIP6
    cmip6_models = load_cmip6_models()
    constraint_cmip6 = constraint_method.train(cmip6_models)
    ecs_dist_cmip6 = constraint_cmip6.estimate_ecs()

    # Compare distributions
    ks_statistic, p_value = scipy.stats.ks_2samp(ecs_dist_cmip5, ecs_dist_cmip6)

    # Consistency metrics
    consistency = {
        'median_shift': np.median(ecs_dist_cmip6) - np.median(ecs_dist_cmip5),
        'range_overlap': calculate_overlap(ecs_dist_cmip5, ecs_dist_cmip6),
        'ks_test': {'statistic': ks_statistic, 'p_value': p_value}
    }

    return consistency
```

### Sensitivity Tests

Test robustness to methodological choices:

```python
def sensitivity_tests(baseline_estimate):
    """
    Test sensitivity to various methodological choices
    """
    sensitivity_results = {}

    # Test 1: Prior choice
    priors = ['uniform', 'jeffreys', 'informed', 'expert']
    for prior in priors:
        ecs = run_analysis(prior=prior)
        sensitivity_results[f'prior_{prior}'] = ecs

    # Test 2: Temperature dataset
    temp_datasets = ['HadCRUT5', 'GISTEMP', 'NOAAGlobalTemp', 'Berkeley']
    for dataset in temp_datasets:
        ecs = run_analysis(temperature_data=dataset)
        sensitivity_results[f'temp_{dataset}'] = ecs

    # Test 3: Model subset
    subsets = ['all_models', 'high_resolution', 'best_historical', 'exclude_outliers']
    for subset in subsets:
        ecs = run_analysis(model_subset=subset)
        sensitivity_results[f'models_{subset}'] = ecs

    # Test 4: Constraint combination
    constraint_sets = [
        ['lgm', 'historical'],
        ['lgm', 'historical', 'cloud'],
        ['all_paleo', 'all_obs', 'all_process']
    ]
    for i, cset in enumerate(constraint_sets):
        ecs = run_analysis(constraints=cset)
        sensitivity_results[f'constraint_set_{i}'] = ecs

    # Analyze sensitivity
    ecs_values = [v['median'] for v in sensitivity_results.values()]
    sensitivity_range = max(ecs_values) - min(ecs_values)

    return sensitivity_results, sensitivity_range
```

## Conservative Reporting

Final estimates reported with full transparency:

```python
def report_final_estimate(ecs_distribution, validation, sensitivity):
    """
    Report ECS estimate with appropriate caveats
    """
    report = {
        # Central estimates
        'median': np.median(ecs_distribution),
        'mean': np.mean(ecs_distribution),

        # Uncertainty ranges
        '66_percent_range': np.percentile(ecs_distribution, [17, 83]),
        '90_percent_range': np.percentile(ecs_distribution, [5, 95]),

        # Validation performance
        'perfect_model_bias': validation['mean_bias'],
        'perfect_model_coverage': validation['coverage_rate'],

        # Sensitivity
        'sensitivity_to_methods': sensitivity['range'],

        # Caveats
        'caveats': [
            'State-dependence not fully quantified',
            'Potential for unknown feedbacks',
            'Model structural uncertainty',
            'Limited by proxy/observational uncertainties'
        ],

        # Confidence assessment
        'confidence_level': assess_confidence(validation, sensitivity)
    }

    return report
```

## References

- Sherwood, S. C., et al. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. Reviews of Geophysics, 58, e2019RG000678.
- IPCC (2021). Climate Change 2021: The Physical Science Basis. Chapter 7: The Earth's Energy Budget, Climate Feedbacks and Climate Sensitivity.
- Tierney, J. E., et al. (2020). Past climates inform our future. Science, 370(6517), eaay3701.
