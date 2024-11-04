import numpy as np
import torch
from visualizer import (
    plot_original_vs_quantized,
    plot_with_quantization_levels,
    plot_distribution,
    plot_quantization_grid,
)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original vector `v`
original_vector = np.array([3.2, -1.4, 2.5, -0.9, 1.8, -3.7, 0.0, 4.0, 2.2, -1.3])

# Uniform Quantization Grid Setup
num_bits = 3
num_levels = 2**num_bits
min_level, max_level = -4.0, 4.0
uniform_quantization_levels = np.linspace(min_level, max_level, num_levels)

# Simple Uniform Quantization
quantized_vector_uniform = np.array(
    [
        uniform_quantization_levels[np.argmin(np.abs(uniform_quantization_levels - x))]
        for x in original_vector
    ]
)
l2_norm_uniform = np.linalg.norm(original_vector - quantized_vector_uniform)
print("L2 Norm (Uniform Quantization):", l2_norm_uniform)

# Visualization for uniform quantization
plot_original_vs_quantized(
    original_vector,
    quantized_vector_uniform,
    title="Original vs. Quantized Vector (Uniform)",
)
plot_with_quantization_levels(
    original_vector,
    uniform_quantization_levels,
    title="Original Vector with Quantization Levels (Uniform)",
)
plot_distribution(
    original_vector,
    uniform_quantization_levels,
    title="Distribution of Original Vector and Quantization Levels (Uniform)",
)
plot_quantization_grid(uniform_quantization_levels, title="Quantization Grid (Uniform)")

# Non-Uniform Quantization - Initializing learnable quantization grid
torch_original_vector = torch.tensor(original_vector, dtype=torch.float32, device=device)
quantization_levels_non_uniform = torch.nn.Parameter(
    torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
)
print("Initial Non-Uniform Quantization Levels:", quantization_levels_non_uniform)

# Optimizer - Using Adam
optimizer = torch.optim.Adam([quantization_levels_non_uniform], lr=0.01)

# Loss Function - Average Distance to Closest Quantization Level
def quantization_loss(v, q_levels):
    expanded_v = v.view(-1, 1)
    distances = (expanded_v - q_levels) ** 2
    min_distances, _ = distances.min(dim=1)
    return min_distances.mean()

# Training Loop for Non-Uniform Quantization
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = quantization_loss(torch_original_vector, quantization_levels_non_uniform)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Final optimized non-uniform quantization levels
optimized_quantization_levels = quantization_levels_non_uniform.detach().cpu().numpy()
print("Optimized Non-Uniform Quantization Levels:", optimized_quantization_levels)

# Loss with Fixed Uniform Grid (for comparison)
uniform_loss = quantization_loss(
    torch_original_vector, torch.tensor(uniform_quantization_levels, device=device)
)
print("Loss with Fixed Uniform Quantization Levels:", uniform_loss.item())

# Quantizing using Optimized Non-Uniform Levels
quantized_vector_non_uniform = np.array(
    [
        optimized_quantization_levels[
            np.argmin(np.abs(optimized_quantization_levels - x))
        ]
        for x in original_vector
    ]
)

# Visualizations for Non-Uniform Quantization
plot_original_vs_quantized(
    original_vector,
    quantized_vector_non_uniform,
    title="Original vs. Quantized Vector (Non-Uniform)",
)
plot_with_quantization_levels(
    original_vector,
    optimized_quantization_levels,
    title="Original Vector with Quantization Levels (Non-Uniform)",
)
plot_distribution(
    original_vector,
    optimized_quantization_levels,
    title="Distribution of Original Vector and Quantization Levels (Non-Uniform)",
)
plot_quantization_grid(
    optimized_quantization_levels, title="Quantization Grid (Non-Uniform)"
)

# Large-scale vector simulation (scaled down from 10 billion for testing)
v_large = np.random.uniform(-4, 4, 10000)

# Simple Uniform Quantization for Large Scale
quantized_vector_large_uniform = np.array(
    [
        uniform_quantization_levels[np.argmin(np.abs(uniform_quantization_levels - x))]
        for x in v_large
    ]
)
l2_norm_large_uniform = np.linalg.norm(v_large - quantized_vector_large_uniform)
print("L2 Norm (Uniform Quantization - Large Scale):", l2_norm_large_uniform)

# Non-Uniform Quantization for Large Scale - Initializing learnable quantization grid
torch_large_vector = torch.tensor(v_large, dtype=torch.float32, device=device)
quantization_levels_large_non_uniform = torch.nn.Parameter(
    torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
)
optimizer_large = torch.optim.Adam([quantization_levels_large_non_uniform], lr=0.01)

# Training Loop for Non-Uniform Quantization on Large Scale
for epoch in range(num_epochs):
    optimizer_large.zero_grad()
    loss_large = quantization_loss(torch_large_vector, quantization_levels_large_non_uniform)
    loss_large.backward()
    optimizer_large.step()

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss (Large): {loss_large.item()}")

# Final optimized non-uniform quantization levels for Large Scale
optimized_levels_large = quantization_levels_large_non_uniform.detach().cpu().numpy()
print("Optimized Non-Uniform Quantization Levels (Large):", optimized_levels_large)

# Quantizing large vector using optimized non-uniform levels
quantized_vector_large_non_uniform = np.array(
    [
        optimized_levels_large[np.argmin(np.abs(optimized_levels_large - x))]
        for x in v_large
    ]
)

# Visualizations for Non-Uniform Quantization on Large Scale
plot_original_vs_quantized(
    v_large,
    quantized_vector_large_non_uniform,
    title="Original vs. Quantized Vector (Non-Uniform) - Large Scale",
)
plot_with_quantization_levels(
    v_large,
    optimized_levels_large,
    title="Original Vector with Quantization Levels (Non-Uniform) - Large Scale",
)
plot_distribution(
    v_large,
    optimized_levels_large,
    title="Distribution of Original Vector and Quantization Levels (Non-Uniform) - Large Scale",
)
plot_quantization_grid(
    optimized_levels_large, title="Quantization Grid (Non-Uniform) - Large Scale"
)



# EasyQuant-inspired implementation additions

def identify_outliers(v, threshold=3.0):
    """ Identify outliers in the vector based on a threshold. """
    mean, std = torch.mean(v), torch.std(v)
    outliers = torch.abs(v - mean) > (threshold * std)
    return outliers

# Create a mask to identify outliers in the original vector
outlier_mask = identify_outliers(torch_original_vector, threshold=3.0)
print(f"Outliers Identified (indices): {torch.nonzero(outlier_mask).cpu().numpy()}")

# Adjust quantization loss function to ignore outliers during optimization
def quantization_loss_with_outliers(v, q_levels, mask):
    """ Compute quantization loss, ignoring outliers based on mask. """
    v_normal = v[~mask]
    expanded_v_normal = v_normal.view(-1, 1)
    distances = (expanded_v_normal - q_levels) ** 2
    min_distances, _ = distances.min(dim=1)
    return min_distances.mean()

# Optimize the quantization range on normal (non-outlier) values only
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = quantization_loss_with_outliers(torch_original_vector, quantization_levels_non_uniform, outlier_mask)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Apply optimized quantization to vector, retaining outliers in full precision
quantized_vector_with_outliers = np.where(
    outlier_mask.cpu().numpy(),
    torch_original_vector.cpu().numpy(),  # Keep outliers in full precision
    optimized_quantization_levels[np.argmin(np.abs(optimized_quantization_levels[:, None] - original_vector), axis=0)]
)


plot_original_vs_quantized(
    original_vector,
    quantized_vector_with_outliers,
    title="Original vs. Quantized Vector (With Outliers in Full Precision)",
)
plot_with_quantization_levels(
    original_vector,
    optimized_quantization_levels,
    title="Original Vector with Quantization Levels (Non-Uniform, Outliers Isolated)",
)
plot_distribution(
    original_vector,
    optimized_quantization_levels,
    title="Distribution of Original Vector and Quantization Levels (Non-Uniform, Outliers Isolated)",
)
plot_quantization_grid(
    optimized_quantization_levels, title="Quantization Grid (Non-Uniform, Outliers Isolated)"
)
