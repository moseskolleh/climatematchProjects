"""
Penman-Monteith Evapotranspiration Module
==========================================

Implements a simplified Penman-Monteith evapotranspiration model
as a differentiable component.
"""

import tensorflow as tf
import numpy as np


class PenmanMonteithET(tf.keras.layers.Layer):
    """
    Simplified Penman-Monteith ET calculation.

    Uses a Priestley-Taylor approximation for computational efficiency
    while maintaining physical basis.

    Inputs:
    -------
    Dictionary with keys:
        - temp: (batch, time_steps) temperature in °C
        - radiation: (batch, time_steps) solar radiation in MJ/m²/day
        - humidity: (batch, time_steps) relative humidity in %
        - LAI: (batch, 1) leaf area index (optional, default 2.0)

    Outputs:
    --------
        - ET: (batch, time_steps) evapotranspiration in mm/day
    """

    def __init__(self, alpha=1.26, gamma=0.067, **kwargs):
        """
        Initialize Penman-Monteith ET module.

        Args:
            alpha: Priestley-Taylor coefficient (default: 1.26)
            gamma: Psychrometric constant in kPa/°C (default: 0.067)
        """
        super(PenmanMonteithET, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, inputs, training=None):
        """
        Calculate evapotranspiration.

        Args:
            inputs: Dictionary of input tensors
            training: Boolean indicating training mode

        Returns:
            ET tensor (batch, time_steps) in mm/day
        """
        temp = inputs['temp']  # °C
        radiation = inputs['radiation']  # MJ/m²/day
        humidity = inputs.get('humidity', None)  # %
        LAI = inputs.get('LAI', tf.constant([[2.0]], dtype=tf.float32))  # Leaf Area Index

        # Calculate saturation vapor pressure (Tetens equation)
        # e_s = 0.6108 * exp(17.27 * T / (T + 237.3))
        e_s = 0.6108 * tf.exp((17.27 * temp) / (temp + 237.3))

        # Slope of saturation vapor pressure curve
        # Delta = 4098 * e_s / (T + 237.3)²
        Delta = 4098.0 * e_s / tf.square(temp + 237.3)

        # Priestley-Taylor equation
        # ET_rad = alpha * (Delta / (Delta + gamma)) * Rn * 0.408
        # 0.408 converts MJ/m²/day to mm/day equivalent evaporation
        ET_rad = self.alpha * (Delta / (Delta + self.gamma)) * radiation * 0.408

        # Vegetation coefficient based on LAI
        # Kc ranges from ~0.3 (bare soil) to ~1.2 (full canopy)
        # Kc = 0.3 + 0.7 * (1 - exp(-0.7 * LAI))
        Kc = 0.3 + 0.7 * (1.0 - tf.exp(-0.7 * LAI))

        # Ensure Kc doesn't exceed reasonable bounds
        Kc = tf.clip_by_value(Kc, 0.3, 1.3)

        # Apply vegetation coefficient
        ET = ET_rad * Kc

        # Ensure non-negative ET
        ET = tf.maximum(ET, 0.0)

        return ET

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Penman-Monteith ET Module")

    # Create sample inputs
    batch_size = 2
    time_steps = 30  # 30 days

    # Typical tropical climate
    temp = tf.constant(np.random.uniform(25, 35, (batch_size, time_steps)), dtype=tf.float32)
    radiation = tf.constant(np.random.uniform(15, 25, (batch_size, time_steps)), dtype=tf.float32)
    humidity = tf.constant(np.random.uniform(60, 90, (batch_size, time_steps)), dtype=tf.float32)
    LAI = tf.constant([[3.0], [1.5]], dtype=tf.float32)

    inputs = {
        'temp': temp,
        'radiation': radiation,
        'humidity': humidity,
        'LAI': LAI
    }

    # Create model and run
    et_model = PenmanMonteithET()
    ET = et_model(inputs)

    print(f"ET shape: {ET.shape}")
    print(f"Mean ET (basin 1): {tf.reduce_mean(ET[0]).numpy():.2f} mm/day")
    print(f"Mean ET (basin 2): {tf.reduce_mean(ET[1]).numpy():.2f} mm/day")
    print(f"ET range: [{tf.reduce_min(ET).numpy():.2f}, {tf.reduce_max(ET).numpy():.2f}] mm/day")
