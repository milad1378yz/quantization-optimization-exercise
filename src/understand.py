import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def plot_original_vs_quantized(v, quantized_vector):
    indices = np.arange(len(v))
    plt.figure(figsize=(12, 6))
    plt.plot(indices, v, 'o-', label='Original Vector', markersize=8)
    plt.plot(indices, quantized_vector, 's--', label='Quantized Vector', markersize=8)
    plt.xticks(indices)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Original vs. Quantized Vector')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_with_quantization_levels(v, quantization_levels):
    indices = np.arange(len(v))
    plt.figure(figsize=(12, 6))
    plt.plot(indices, v, 'o', label='Original Vector', markersize=8)
    for level in quantization_levels:
        plt.hlines(level, xmin=-0.5, xmax=len(v)-0.5, colors='gray', linestyles='dashed')
    plt.xticks(indices)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Original Vector with Quantization Levels')
    plt.legend(['Original Vector', 'Quantization Levels'])
    plt.grid(True)
    plt.show()

def plot_distribution(v, quantization_levels):
    plt.figure(figsize=(10, 6))
    plt.hist(v, bins=10, alpha=0.5, label='Original Vector Distribution')
    plt.scatter(quantization_levels, [0] * len(quantization_levels), color='red', label='Quantization Levels', zorder=3)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Original Vector and Quantization Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_quantization_grid(quantization_levels):
    plt.figure(figsize=(10, 6))
    plt.scatter(quantization_levels, np.zeros_like(quantization_levels), color='blue', s=100, label='Quantization Levels')
    plt.yticks([])
    plt.xlabel('Quantization Level')
    plt.title('Quantization Grid')
    plt.legend()
    plt.grid(True)
    plt.show()

# Running all plots
plot_original_vs_quantized(v, quantized_vector)
plot_with_quantization_levels(v, quantization_levels)
plot_distribution(v, quantization_levels)
plot_quantization_grid(quantization_levels)
