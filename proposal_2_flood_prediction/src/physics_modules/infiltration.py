"""
Green-Ampt Infiltration Module
===============================

Implements the Green-Ampt infiltration model as a differentiable component.

The Green-Ampt model calculates infiltration rate based on:
- Hydraulic conductivity (K)
- Wetting front suction head (psi)
- Initial moisture deficit (delta_theta)
"""

import tensorflow as tf
import numpy as np


class GreenAmptInfiltration(tf.keras.layers.Layer):
    """
    Green-Ampt infiltration model implemented as a differentiable Keras layer.

    Parameters:
    -----------
    dt : float
        Time step in hours (default: 1.0)
    min_infiltration : float
        Minimum infiltration rate in mm/hr (default: 0.1)

    Inputs:
    -------
    Dictionary with keys:
        - precip: (batch, time_steps) precipitation in mm/hr
        - K: (batch, 1) hydraulic conductivity in mm/hr
        - psi: (batch, 1) wetting front suction head in mm
        - delta_theta: (batch, 1) initial moisture deficit (dimensionless)

    Outputs:
    --------
    Dictionary with keys:
        - infiltration: (batch, time_steps) infiltration rate in mm/hr
        - runoff: (batch, time_steps) surface runoff in mm/hr
        - cumulative_infiltration: (batch, time_steps) cumulative infiltration in mm
    """

    def __init__(self, dt=1.0, min_infiltration=0.1, **kwargs):
        super(GreenAmptInfiltration, self).__init__(**kwargs)
        self.dt = dt  # Time step in hours
        self.min_infiltration = min_infiltration

    def call(self, inputs, training=None):
        """
        Forward pass through the Green-Ampt infiltration model.

        Args:
            inputs: Dictionary of input tensors
            training: Boolean indicating training mode

        Returns:
            Dictionary of output tensors
        """
        precip = inputs['precip']  # (batch, time_steps)
        K = inputs['K']  # (batch, 1)
        psi = inputs['psi']  # (batch, 1)
        delta_theta = inputs['delta_theta']  # (batch, 1)

        # Initialize cumulative infiltration
        F = tf.zeros_like(precip[:, 0:1])  # (batch, 1)

        infiltration_list = []
        runoff_list = []
        F_list = []

        # Iterate through time steps
        for t in range(precip.shape[1]):
            P_t = precip[:, t:t+1]  # (batch, 1)

            # Calculate infiltration capacity using Green-Ampt equation
            # f = K * (1 + (psi * delta_theta) / F)
            # Prevent division by zero with small epsilon
            epsilon = 1e-6
            f_capacity = K * (1.0 + (psi * delta_theta) / tf.maximum(F, epsilon))

            # Ensure minimum infiltration rate
            f_capacity = tf.maximum(f_capacity, self.min_infiltration)

            # Actual infiltration is minimum of capacity and precipitation
            f_actual = tf.minimum(P_t, f_capacity)

            # Surface runoff is excess precipitation
            runoff_t = tf.maximum(P_t - f_actual, 0.0)

            # Update cumulative infiltration
            F = F + f_actual * self.dt

            # Store results
            infiltration_list.append(f_actual)
            runoff_list.append(runoff_t)
            F_list.append(F)

        # Stack results along time dimension
        infiltration = tf.concat(infiltration_list, axis=1)
        runoff = tf.concat(runoff_list, axis=1)
        cumulative_infiltration = tf.concat(F_list, axis=1)

        return {
            'infiltration': infiltration,
            'runoff': runoff,
            'cumulative_infiltration': cumulative_infiltration
        }

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'dt': self.dt,
            'min_infiltration': self.min_infiltration
        })
        return config


# Testing and validation functions
def validate_green_ampt_analytical(K=10.0, psi=100.0, delta_theta=0.3, precip=20.0, duration=10):
    """
    Validate Green-Ampt implementation against analytical solution.

    For constant rainfall intensity, the Green-Ampt model can be solved
    analytically for ponding time and subsequent infiltration.

    Args:
        K: Hydraulic conductivity (mm/hr)
        psi: Suction head (mm)
        delta_theta: Moisture deficit
        precip: Constant precipitation rate (mm/hr)
        duration: Simulation duration (hours)

    Returns:
        Dictionary with validation results
    """
    # TODO: Implement analytical solution comparison
    pass


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Green-Ampt Infiltration Module")

    # Create sample inputs
    batch_size = 2
    time_steps = 24  # 24 hours

    # Constant precipitation
    precip = tf.constant(np.ones((batch_size, time_steps)) * 15.0, dtype=tf.float32)

    # Soil parameters (typical values)
    K = tf.constant([[10.0], [5.0]], dtype=tf.float32)  # mm/hr
    psi = tf.constant([[100.0], [150.0]], dtype=tf.float32)  # mm
    delta_theta = tf.constant([[0.3], [0.25]], dtype=tf.float32)  # dimensionless

    inputs = {
        'precip': precip,
        'K': K,
        'psi': psi,
        'delta_theta': delta_theta
    }

    # Create model and run
    infiltration_model = GreenAmptInfiltration(dt=1.0)
    outputs = infiltration_model(inputs)

    print(f"Infiltration shape: {outputs['infiltration'].shape}")
    print(f"Runoff shape: {outputs['runoff'].shape}")
    print(f"Cumulative infiltration shape: {outputs['cumulative_infiltration'].shape}")

    print(f"\nSample infiltration (first basin, first 5 hours): {outputs['infiltration'][0, :5].numpy()}")
    print(f"Sample runoff (first basin, first 5 hours): {outputs['runoff'][0, :5].numpy()}")
    print(f"Final cumulative infiltration: {outputs['cumulative_infiltration'][0, -1].numpy():.2f} mm")
