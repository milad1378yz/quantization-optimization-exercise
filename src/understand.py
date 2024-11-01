import numpy as np

from scipy.optimize import minimize

from visualizer import plot_original_vs_quantized, plot_with_quantization_levels, plot_distribution, plot_quantization_grid
# Original vector
v = np.array([3.2, -1.4, 2.5, -0.9, 1.8, -3.7, 0.0, 4.0, 2.2, -1.3])

# Quantization grid setup
num_bits = 3
num_levels = 2 ** num_bits
min_level = -4.0
max_level = 4.0
quantization_levels = np.linspace(min_level, max_level, num_levels)

quantized_vector = np.array([quantization_levels[np.argmin(np.abs(quantization_levels - x))] for x in v])
l2_norm = np.linalg.norm(v - quantized_vector)



# Running all plots
plot_original_vs_quantized(v, quantized_vector)
plot_with_quantization_levels(v, quantization_levels)
plot_distribution(v, quantization_levels)
plot_quantization_grid(quantization_levels)
