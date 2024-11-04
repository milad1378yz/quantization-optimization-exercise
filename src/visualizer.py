import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")
import os

os.makedirs("images", exist_ok=True)


def plot_original_vs_quantized(
    v, quantized_vector, title="Original vs. Quantized Vector"
):
    """Plot the original vector and the quantized vector side by side"""
    indices = np.arange(len(v))
    plt.figure(figsize=(12, 6))
    plt.plot(indices, v, "o-", label="Original Vector", markersize=8)
    plt.plot(indices, quantized_vector, "s--", label="Quantized Vector", markersize=8)
    plt.xticks(indices)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/{title.lower().replace(' ', '_').replace('.', '_')}.png")


def plot_with_quantization_levels(
    v, quantization_levels, title="Original Vector with Quantization Levels"
):
    """Plot the original vector with the quantization levels"""
    indices = np.arange(len(v))
    plt.figure(figsize=(12, 6))
    plt.plot(indices, v, "o", label="Original Vector", markersize=8)
    for level in quantization_levels:
        plt.hlines(
            level, xmin=-0.5, xmax=len(v) - 0.5, colors="gray", linestyles="dashed"
        )
    plt.xticks(indices)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(["Original Vector", "Quantization Levels"])
    plt.grid(True)
    plt.savefig(f"images/{title.lower().replace(' ', '_').replace('.', '_')}.png")


def plot_distribution(
    v,
    quantization_levels,
    title="Distribution of Original Vector and Quantization Levels",
):
    """Plot the distribution of the original vector and the quantization levels"""
    plt.figure(figsize=(10, 6))
    plt.hist(v, bins=10, alpha=0.5, label="Original Vector Distribution")
    plt.scatter(
        quantization_levels,
        [0] * len(quantization_levels),
        color="red",
        label="Quantization Levels",
        zorder=3,
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/{title.lower().replace(' ', '_').replace('.', '_')}.png")


def plot_quantization_grid(quantization_levels, title="Quantization Grid"):
    """Plot the quantization grid"""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        quantization_levels,
        np.zeros_like(quantization_levels),
        color="blue",
        s=100,
        label="Quantization Levels",
    )
    plt.yticks([])
    plt.xlabel("Quantization Level")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/{title.lower().replace(' ', '_').replace('.', '_')}.png")
