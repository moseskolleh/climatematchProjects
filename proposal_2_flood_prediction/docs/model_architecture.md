# Hybrid Model Architecture: Technical Specification

## Architecture Overview

The Hybrid Physics-ML Framework integrates differentiable physics-based hydrological modules with neural network components to create a unified, end-to-end trainable flood prediction system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT DATA LAYER                              │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ Precipitation │  │ Temperature  │  │ Basin Characteristics   │  │
│  │ (Satellite)   │  │ (Reanalysis) │  │ (DEM, Soil, Land Cover) │  │
│  └───────┬───────┘  └──────┬───────┘  └───────────┬─────────────┘  │
└──────────┼──────────────────┼──────────────────────┼────────────────┘
           │                  │                      │
           ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PREPROCESSING LAYER                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Precipitation Downscaling Network (CNN)                   │    │
│  │  Input: Coarse satellite precip → Output: Fine-scale precip│    │
│  └────────────────────────────────┬───────────────────────────┘    │
│                                   │                                 │
│  ┌────────────────────────────────▼───────────────────────────┐    │
│  │  Parameter Estimation Network (MLP/RF)                     │    │
│  │  Input: Basin features → Output: Physics model parameters  │    │
│  └────────────────────────────────┬───────────────────────────┘    │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHYSICS MODULES (DIFFERENTIABLE)                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  1. Infiltration Module (Green-Ampt)                       │    │
│  │     - Compute infiltration from precipitation              │    │
│  │     - Update soil moisture states                          │    │
│  │     - Generate surface runoff                              │    │
│  └────────────────────────┬───────────────────────────────────┘    │
│                           │                                          │
│  ┌────────────────────────▼───────────────────────────────────┐    │
│  │  2. Evapotranspiration Module (Penman-Monteith)            │    │
│  │     - Calculate potential ET                                │    │
│  │     - Apply crop/vegetation coefficients                   │    │
│  │     - Update water balance                                 │    │
│  └────────────────────────┬───────────────────────────────────┘    │
│                           │                                          │
│  ┌────────────────────────▼───────────────────────────────────┐    │
│  │  3. Routing Module (Kinematic Wave)                        │    │
│  │     - Route runoff through river network                   │    │
│  │     - Solve Saint-Venant equations (simplified)            │    │
│  │     - Compute discharge at outlet                          │    │
│  └────────────────────────┬───────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML STATE CORRECTION LAYER                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LSTM State Correction Network                             │    │
│  │  - Inputs: Physics states, recent observations             │    │
│  │  - Outputs: Correction factors                             │    │
│  │  - Apply multiplicative/additive corrections               │    │
│  └────────────────────────────────┬───────────────────────────┘    │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Probabilistic Discharge Forecast                          │    │
│  │  - Mean prediction                                         │    │
│  │  - Uncertainty bounds (quantiles)                          │    │
│  │  - Lead times: 24h, 48h, 72h                              │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. ML Preprocessing Layer

### 1.1 Precipitation Downscaling Network

**Architecture**: U-Net style Convolutional Neural Network

**Purpose**: Convert coarse-resolution satellite precipitation to fine-scale basin-specific precipitation

**Layers**:
```python
Input: (batch, time, lat, lon, channels=1)  # Coarse precip
       + (batch, lat_fine, lon_fine, features) # Static features (elevation, aspect)

Encoder:
  Conv2D(64, kernel=3, padding='same') + ReLU
  Conv2D(64, kernel=3, padding='same') + ReLU
  MaxPool2D(2)

  Conv2D(128, kernel=3, padding='same') + ReLU
  Conv2D(128, kernel=3, padding='same') + ReLU
  MaxPool2D(2)

  Conv2D(256, kernel=3, padding='same') + ReLU
  Conv2D(256, kernel=3, padding='same') + ReLU

Decoder (with skip connections):
  UpSample2D(2) + Conv2D(128, kernel=3) + ReLU
  Concatenate with encoder layer
  Conv2D(128, kernel=3) + ReLU

  UpSample2D(2) + Conv2D(64, kernel=3) + ReLU
  Concatenate with encoder layer
  Conv2D(64, kernel=3) + ReLU

Output: Conv2D(1, kernel=1) + ReLU  # Fine-scale precip
```

**Loss Function**:
- Mean Squared Error (MSE) for continuous prediction
- Binary Cross-Entropy for rain/no-rain classification (auxiliary task)
- Spatial gradient penalty to ensure realistic spatial patterns

**Training Data**:
- High-resolution radar or dense gauge network data (from data-rich regions)
- Transfer to West Africa using domain adaptation

### 1.2 Parameter Estimation Network

**Architecture**: Multi-Layer Perceptron (MLP) or Random Forest

**Purpose**: Estimate physics model parameters from basin characteristics

**Input Features** (per basin):
- Area (km²)
- Mean elevation (m)
- Mean slope (degrees)
- Drainage density (km/km²)
- Soil texture fractions (% sand, silt, clay)
- Land cover fractions (% forest, agriculture, urban, etc.)
- Climate indices (mean annual precip, aridity index)
- Latitude, longitude

**Output Parameters**:
- Manning's roughness coefficient (n)
- Hydraulic conductivity (mm/hr)
- Wetting front suction head (mm)
- Initial soil moisture deficit (mm)
- Channel geometry (width, depth as power-law parameters)

**MLP Architecture**:
```python
Input: (batch, features=15)

Dense(128) + ReLU + Dropout(0.2)
Dense(64) + ReLU + Dropout(0.2)
Dense(32) + ReLU
Dense(num_params) + Custom activation to ensure physical ranges

Output: (batch, num_params=8)
```

**Custom Activation**:
- Sigmoid scaled to physical parameter ranges
- Example: Manning's n in [0.01, 0.15], conductivity in [0.1, 100] mm/hr

**Training**:
- Train on basins with known parameters (calibrated from observations)
- Loss: MSE between predicted and calibrated parameters
- Regularization: L2 to prevent overfitting

## 2. Physics Modules (Differentiable Implementation)

### 2.1 Green-Ampt Infiltration Module

**Governing Equation**:
```
f(t) = K * (1 + (ψ * Δθ) / F(t))

Where:
f(t) = infiltration rate at time t
K = hydraulic conductivity
ψ = wetting front suction head
Δθ = initial moisture deficit
F(t) = cumulative infiltration
```

**Differentiable Implementation** (TensorFlow/PyTorch):

```python
class GreenAmptInfiltration(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """
        inputs: dict with keys
          - precip: (batch, time_steps) [mm/hr]
          - K: (batch, 1) hydraulic conductivity [mm/hr]
          - psi: (batch, 1) suction head [mm]
          - delta_theta: (batch, 1) moisture deficit [-]

        outputs: dict with keys
          - infiltration: (batch, time_steps) [mm/hr]
          - runoff: (batch, time_steps) [mm/hr]
          - soil_moisture: (batch, time_steps) [mm]
        """
        precip = inputs['precip']
        K = inputs['K']
        psi = inputs['psi']
        delta_theta = inputs['delta_theta']

        # Initialize cumulative infiltration
        F = tf.zeros_like(precip[:, 0:1])

        infiltration_list = []
        runoff_list = []

        for t in range(precip.shape[1]):
            P_t = precip[:, t:t+1]

            # Infiltration capacity
            f_cap = K * (1 + (psi * delta_theta) / tf.maximum(F, 1e-6))

            # Actual infiltration (limited by precipitation and capacity)
            f_actual = tf.minimum(P_t, f_cap)

            # Surface runoff
            runoff_t = tf.maximum(P_t - f_actual, 0.0)

            # Update cumulative infiltration
            F = F + f_actual * dt  # dt = time step in hours

            infiltration_list.append(f_actual)
            runoff_list.append(runoff_t)

        infiltration = tf.stack(infiltration_list, axis=1)
        runoff = tf.stack(runoff_list, axis=1)

        return {'infiltration': infiltration, 'runoff': runoff}
```

**Trainable Parameters**: K, ψ, Δθ (estimated from Parameter Estimation Network)

**Gradient Flow**: Fully differentiable, gradients flow back through parameter estimation

### 2.2 Penman-Monteith Evapotranspiration Module

**Simplified Equation**:
```
ET = (Δ * (Rn - G) + ρ * cp * VPD / ra) / (Δ + γ * (1 + rs/ra))

Where:
Δ = slope of saturation vapor pressure curve
Rn = net radiation
G = ground heat flux
ρ = air density
cp = specific heat of air
VPD = vapor pressure deficit
ra = aerodynamic resistance
rs = surface resistance
γ = psychrometric constant
```

**Simplified Implementation**:
```python
class PenmanMonteithET(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        inputs: dict
          - temp: (batch, time_steps) temperature [°C]
          - radiation: (batch, time_steps) solar radiation [MJ/m²/day]
          - humidity: (batch, time_steps) relative humidity [%]
          - wind_speed: (batch, time_steps) wind speed [m/s]
          - LAI: (batch, 1) leaf area index

        outputs:
          - ET: (batch, time_steps) evapotranspiration [mm/day]
        """
        temp = inputs['temp']
        rad = inputs['radiation']
        RH = inputs['humidity']
        u = inputs['wind_speed']
        LAI = inputs['LAI']

        # Saturation vapor pressure
        e_s = 0.6108 * tf.exp((17.27 * temp) / (temp + 237.3))

        # Actual vapor pressure
        e_a = e_s * RH / 100.0

        # Vapor pressure deficit
        VPD = e_s - e_a

        # Slope of saturation vapor pressure curve
        Delta = 4098 * e_s / tf.square(temp + 237.3)

        # Simplified PM equation (Priestley-Taylor approximation)
        alpha = 1.26  # Priestley-Taylor coefficient
        gamma = 0.067  # Psychrometric constant [kPa/°C]

        ET_rad = alpha * (Delta / (Delta + gamma)) * rad * 0.408

        # Apply vegetation coefficient based on LAI
        Kc = tf.minimum(0.3 + 0.7 * (1 - tf.exp(-0.7 * LAI)), 1.3)

        ET = ET_rad * Kc

        return ET
```

### 2.3 Kinematic Wave Routing Module

**Governing Equations**:
```
Continuity: ∂Q/∂x + ∂A/∂t = q_lat
Momentum (simplified): Q = α * A^β  (Manning's equation)

Where:
Q = discharge [m³/s]
A = cross-sectional area [m²]
q_lat = lateral inflow [m³/s/m]
α, β = channel geometry parameters
```

**Finite Difference Implementation**:
```python
class KinematicWaveRouting(tf.keras.layers.Layer):
    def __init__(self, river_length, num_segments=10):
        super().__init__()
        self.L = river_length  # meters
        self.dx = river_length / num_segments
        self.num_seg = num_segments

    def call(self, inputs):
        """
        inputs: dict
          - runoff: (batch, time_steps) lateral inflow [mm/hr]
          - basin_area: (batch, 1) basin area [km²]
          - manning_n: (batch, 1) roughness coefficient
          - slope: (batch, 1) channel slope [m/m]
          - width: (batch, 1) channel width [m]

        outputs:
          - discharge: (batch, time_steps) discharge at outlet [m³/s]
        """
        runoff = inputs['runoff']  # mm/hr
        area = inputs['basin_area'] * 1e6  # convert to m²
        n = inputs['manning_n']
        S = inputs['slope']
        W = inputs['width']

        # Convert runoff to m³/s
        q_lat = (runoff / 1000 / 3600) * area / self.num_seg

        # Manning's equation parameters
        # Q = (1/n) * A^(5/3) / P^(2/3) * sqrt(S)
        # For wide rectangular channel: A ≈ W*h, P ≈ W
        # Simplification: Q = alpha * h^beta
        alpha = (1/n) * W**(2/3) * tf.sqrt(S)
        beta = 5/3

        # Initialize discharge array
        Q = tf.zeros((inputs['runoff'].shape[0], self.num_seg + 1))
        discharge_outlet = []

        dt = 3600.0  # time step in seconds (1 hour)
        c = alpha * beta  # wave celerity approximation

        for t in range(runoff.shape[1]):
            q_in = q_lat[:, t:t+1]

            # Upstream boundary
            Q_new = Q[:, :1]

            # Route through segments
            for i in range(1, self.num_seg + 1):
                # Upwind scheme
                Q_new_i = Q[:, i:i+1] + (dt/self.dx) * (Q[:, i-1:i] - Q[:, i:i+1]) + q_in * dt
                Q_new_i = tf.maximum(Q_new_i, 0.0)  # Ensure non-negative
                Q_new = tf.concat([Q_new, Q_new_i], axis=1)

            Q = Q_new
            discharge_outlet.append(Q[:, -1:])

        discharge_out = tf.concat(discharge_outlet, axis=1)

        return discharge_out
```

**Numerical Stability**:
- CFL condition checked: c * dt / dx < 1
- Adaptive time-stepping if needed

## 3. ML State Correction Layer

### 3.1 LSTM State Correction Network

**Purpose**: Learn systematic biases in physics model and correct states

**Architecture**:
```python
Input: (batch, time_steps, features)
  features = [
    discharge_physics,      # Physics model output
    soil_moisture_physics,  # Physics model state
    precip_cumulative,      # Recent precipitation sum
    obs_recent,            # Recent observations
    day_of_year,           # Seasonal indicator
    antecedent_conditions  # Prior wetness
  ]

LSTM(128, return_sequences=True)
LSTM(64, return_sequences=True)
Dense(32) + ReLU
Dense(num_correction_factors)

Output: Correction factors (multiplicative and additive)
```

**Application of Corrections**:
```python
discharge_corrected = discharge_physics * correction_mult + correction_add
```

**Training**:
- Minimize difference between corrected output and observations
- Regularization to prevent over-correction (keep close to physics)

## 4. Ensemble for Uncertainty Quantification

### 4.1 Ensemble Generation Methods

**Method 1: Deep Ensemble**
- Train N=10 models with different random initializations
- Average predictions: μ = mean(y_1, ..., y_N)
- Uncertainty: σ² = variance(y_1, ..., y_N) + mean(σ_i²)

**Method 2: MC Dropout**
- Apply dropout during inference (p=0.1-0.2)
- Generate M=100 stochastic forward passes
- Predictive distribution from samples

**Method 3: Input Perturbations**
- Perturb precipitation within uncertainty bounds
- Perturb parameters within estimated ranges
- Ensemble from perturbed inputs

### 4.2 Probabilistic Output

```python
Output for each time step:
  - mean_discharge: E[Q(t)]
  - std_discharge: sqrt(Var[Q(t)])
  - quantiles: [Q_0.1(t), Q_0.25(t), Q_0.5(t), Q_0.75(t), Q_0.9(t)]
```

## 5. Loss Function Design

### 5.1 Multi-Component Loss

```python
L_total = w1 * L_data + w2 * L_physics + w3 * L_reg

Where:
  L_data: Data fidelity (MSE, MAE, or quantile loss)
  L_physics: Physics constraint violations
  L_reg: Regularization (parameter smoothness, temporal consistency)
```

### 5.2 Data Loss

**For Deterministic Prediction**:
```python
L_data = MSE(Q_pred, Q_obs) + λ * MAE(log(Q_pred + ε), log(Q_obs + ε))
```
The log-space term ensures good performance on low flows.

**For Probabilistic Prediction** (Quantile Loss):
```python
For quantile τ:
L_quantile = Σ_t ρ_τ(Q_obs(t) - Q_τ(t))

Where ρ_τ(u) = u * (τ - 1{u < 0})
```

### 5.3 Physics Loss

**Mass Conservation**:
```python
L_mass = MSE(cumulative_precip - cumulative_ET - cumulative_runoff - Δstorage, 0)
```

**Non-Negativity Constraints**:
```python
L_nonneg = ReLU(-soil_moisture) + ReLU(-discharge) + ReLU(-infiltration)
```

**Parameter Range Constraints**:
```python
L_param = ReLU(K - K_max) + ReLU(K_min - K)  # for all parameters
```

## 6. Training Procedure

### 6.1 Two-Stage Training

**Stage 1: Physics Module Warm-up**
- Fix ML components, train physics parameters
- Minimize data loss only
- Ensures physics modules are in reasonable range

**Stage 2: End-to-End Fine-tuning**
- Jointly train all components
- Use full multi-component loss
- Apply gradual unfreezing if needed

### 6.2 Training Algorithm

```
Initialize parameters
For epoch in 1:num_epochs:
  For each basin in training set:
    # Forward pass
    precip_fine = downscaling_network(precip_coarse, static_features)
    params = parameter_network(basin_characteristics)

    runoff = infiltration_module(precip_fine, params)
    ET = evapotranspiration_module(met_data, LAI)
    discharge_physics = routing_module(runoff, params)

    discharge_corrected = lstm_correction(discharge_physics, states, obs_recent)

    # Calculate losses
    L_data = data_loss(discharge_corrected, observations)
    L_physics = physics_loss(runoff, ET, precip, storage)
    L_reg = regularization_loss(params)

    L_total = w1*L_data + w2*L_physics + w3*L_reg

    # Backward pass
    gradients = compute_gradients(L_total, all_parameters)
    optimizer.apply_gradients(gradients)

  # Validation
  val_loss = evaluate_on_validation_set()
  if val_loss improved:
    save_checkpoint()

  # Learning rate schedule
  adjust_learning_rate()
```

### 6.3 Hyperparameters

- **Learning rate**: 1e-3 initially, decay to 1e-5
- **Batch size**: 16 basins
- **Epochs**: 100-200 (with early stopping)
- **Optimizer**: Adam
- **Loss weights**: w1=1.0, w2=0.1, w3=0.01 (tune via validation)

## 7. Inference (Operational Forecasting)

### 7.1 Real-Time Prediction Pipeline

```
1. Acquire latest data:
   - Download latest satellite precipitation
   - Fetch meteorological forecasts (ECMWF)
   - Get upstream gauge readings

2. Preprocess:
   - Apply quality control
   - Downscale precipitation

3. Initialize model:
   - Load latest model checkpoint
   - Initialize states from last run (or spin-up)

4. Generate ensemble:
   - Run ensemble members (N=10)
   - Aggregate into probabilistic forecast

5. Post-process:
   - Apply bias correction if needed
   - Compute flood probabilities (exceedance of thresholds)

6. Disseminate:
   - Generate forecast bulletin
   - Send alerts if thresholds exceeded
   - Update web dashboard
```

### 7.2 Computational Requirements

**Training**:
- GPU: NVIDIA RTX 3090 or better
- RAM: 32 GB
- Storage: 1 TB SSD
- Time: ~48 hours for full training

**Inference** (per basin, per forecast):
- CPU: 4 cores sufficient
- RAM: 4 GB
- Time: <5 minutes for 72-hour forecast

---

*This architecture will be implemented modularly to allow component-wise testing and iterative refinement.*
