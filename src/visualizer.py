import matplotlib.pyplot as plt
import numpy as np


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