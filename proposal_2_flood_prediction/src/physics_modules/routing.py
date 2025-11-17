"""
Kinematic Wave Routing Module
==============================

Implements kinematic wave routing for river networks using finite difference methods.
"""

import tensorflow as tf
import numpy as np


class KinematicWaveRouting(tf.keras.layers.Layer):
    """
    Kinematic wave routing for river discharge.

    Solves the kinematic wave approximation of the Saint-Venant equations
    using an upwind finite difference scheme.

    Parameters:
    -----------
    river_length : float
        Total river length in meters
    num_segments : int
        Number of spatial segments for discretization
    dt : float
        Time step in seconds (default: 3600 for 1 hour)

    Inputs:
    -------
    Dictionary with keys:
        - runoff: (batch, time_steps) lateral inflow in mm/hr
        - basin_area: (batch, 1) basin area in km²
        - manning_n: (batch, 1) Manning's roughness coefficient
        - slope: (batch, 1) channel slope in m/m
        - width: (batch, 1) channel width in meters

    Outputs:
    --------
        - discharge: (batch, time_steps) discharge at outlet in m³/s
    """

    def __init__(self, river_length=50000.0, num_segments=20, dt=3600.0, **kwargs):
        super(KinematicWaveRouting, self).__init__(**kwargs)
        self.river_length = river_length  # meters
        self.num_segments = num_segments
        self.dx = river_length / num_segments  # spatial step
        self.dt = dt  # time step (seconds)

    def call(self, inputs, training=None):
        """
        Route runoff through river network.

        Args:
            inputs: Dictionary of input tensors
            training: Boolean indicating training mode

        Returns:
            discharge: Tensor (batch, time_steps) in m³/s
        """
        runoff = inputs['runoff']  # mm/hr
        basin_area = inputs['basin_area']  # km²
        manning_n = inputs['manning_n']  # dimensionless
        slope = inputs['slope']  # m/m
        width = inputs['width']  # meters

        batch_size = tf.shape(runoff)[0]
        time_steps = tf.shape(runoff)[1]

        # Convert runoff from mm/hr to m³/s per segment
        # runoff [mm/hr] * area [km²] = mm*km²/hr
        # = 1e-3 m * 1e6 m² / 3600 s = (1e3/3600) m³/s
        area_m2 = basin_area * 1e6  # Convert km² to m²
        q_lateral = (runoff / 1000.0 / 3600.0) * area_m2 / float(self.num_segments)

        # Manning's equation for wave celerity
        # For wide rectangular channel: Q = (1/n) * W^(2/3) * h^(5/3) * sqrt(S)
        # Simplified celerity: c ≈ (5/3) * (1/n) * W^(2/3) * sqrt(S)
        # This is an approximation for initial estimate

        # Initialize discharge array (batch, num_segments + 1)
        Q = tf.zeros((batch_size, self.num_segments + 1), dtype=tf.float32)

        discharge_outlet = []

        # Time stepping
        for t in range(time_steps):
            q_in = q_lateral[:, t:t+1]  # (batch, 1)

            # Create new discharge array
            Q_new_list = [Q[:, 0:1]]  # Upstream boundary condition (Q=0)

            # Spatial stepping (upwind scheme)
            for i in range(1, self.num_segments + 1):
                # Simple upwind scheme: dQ/dt = -c * dQ/dx + q_lateral
                # Q_new = Q_old - (c * dt / dx) * (Q_old[i] - Q_old[i-1]) + q_in * dt

                # Estimate wave celerity (simplified)
                # For stability, limit celerity
                c = 2.0  # m/s (simplified, could be made more sophisticated)

                # CFL condition for stability
                cfl = c * self.dt / self.dx
                cfl = tf.minimum(cfl, 0.9)  # Ensure stability

                # Upwind scheme
                Q_i_old = Q[:, i:i+1]
                Q_i_minus1_old = Q[:, i-1:i]

                Q_i_new = Q_i_old - cfl * (Q_i_old - Q_i_minus1_old) + q_in * self.dt

                # Ensure non-negative discharge
                Q_i_new = tf.maximum(Q_i_new, 0.0)

                Q_new_list.append(Q_i_new)

            # Update discharge array
            Q = tf.concat(Q_new_list, axis=1)

            # Store outlet discharge
            discharge_outlet.append(Q[:, -1:])

        # Stack outlet discharges over time
        discharge = tf.concat(discharge_outlet, axis=1)

        return discharge

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'river_length': self.river_length,
            'num_segments': self.num_segments,
            'dt': self.dt
        })
        return config


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Kinematic Wave Routing Module")

    # Create sample inputs
    batch_size = 2
    time_steps = 48  # 48 hours

    # Simulate a rainfall event (peak at hour 10-15)
    runoff = np.zeros((batch_size, time_steps))
    runoff[:, 10:15] = 20.0  # mm/hr during peak
    runoff[:, 8:10] = 5.0
    runoff[:, 15:18] = 10.0
    runoff = tf.constant(runoff, dtype=tf.float32)

    # Basin parameters
    basin_area = tf.constant([[500.0], [1000.0]], dtype=tf.float32)  # km²
    manning_n = tf.constant([[0.035], [0.045]], dtype=tf.float32)
    slope = tf.constant([[0.001], [0.0005]], dtype=tf.float32)  # m/m
    width = tf.constant([[50.0], [80.0]], dtype=tf.float32)  # meters

    inputs = {
        'runoff': runoff,
        'basin_area': basin_area,
        'manning_n': manning_n,
        'slope': slope,
        'width': width
    }

    # Create model and run
    routing_model = KinematicWaveRouting(river_length=50000.0, num_segments=20, dt=3600.0)
    discharge = routing_model(inputs)

    print(f"Discharge shape: {discharge.shape}")
    print(f"Peak discharge (basin 1): {tf.reduce_max(discharge[0]).numpy():.2f} m³/s")
    print(f"Peak discharge (basin 2): {tf.reduce_max(discharge[1]).numpy():.2f} m³/s")
    print(f"Time to peak (basin 1): {tf.argmax(discharge[0]).numpy()} hours after start")
    print(f"Time to peak (basin 2): {tf.argmax(discharge[1]).numpy()} hours after start")
