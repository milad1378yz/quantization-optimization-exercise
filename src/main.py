import numpy as np
import torch
from visualizer import plot_original_vs_quantized, plot_with_quantization_levels, plot_distribution, plot_quantization_grid

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
num_levels = 2 ** num_bits
min_level, max_level = -4.0, 4.0
uniform_quantization_levels = np.linspace(min_level, max_level, num_levels)

# Simple Uniform Quantization
quantized_vector_uniform = np.array([
    uniform_quantization_levels[np.argmin(np.abs(uniform_quantization_levels - x))]
    for x in original_vector
])
l2_norm_uniform = np.linalg.norm(original_vector - quantized_vector_uniform)
print("L2 Norm (Uniform Quantization):", l2_norm_uniform)

# Visualization for uniform quantization
plot_original_vs_quantized(original_vector, quantized_vector_uniform, title='Original vs. Quantized Vector (Uniform)')
plot_with_quantization_levels(original_vector, uniform_quantization_levels, title='Original Vector with Quantization Levels (Uniform)')
plot_distribution(original_vector, uniform_quantization_levels, title='Distribution of Original Vector and Quantization Levels (Uniform)')
plot_quantization_grid(uniform_quantization_levels, title='Quantization Grid (Uniform)')

# Non-Uniform Quantization - Initializing learnable quantization grid
torch_original_vector = torch.tensor(original_vector, dtype=torch.float32, device=device)
quantization_levels_non_uniform = torch.nn.Parameter(torch.rand(num_levels, device=device) * (max_level - min_level) + min_level)
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
uniform_loss = quantization_loss(torch_original_vector, torch.tensor(uniform_quantization_levels, device=device))
print("Loss with Fixed Uniform Quantization Levels:", uniform_loss.item())

# Quantizing using Optimized Non-Uniform Levels
quantized_vector_non_uniform = np.array([
    optimized_quantization_levels[np.argmin(np.abs(optimized_quantization_levels - x))]
    for x in original_vector
])

# Visualizations for Non-Uniform Quantization
plot_original_vs_quantized(original_vector, quantized_vector_non_uniform, title='Original vs. Quantized Vector (Non-Uniform)')
plot_with_quantization_levels(original_vector, optimized_quantization_levels, title='Original Vector with Quantization Levels (Non-Uniform)')
plot_distribution(original_vector, optimized_quantization_levels, title='Distribution of Original Vector and Quantization Levels (Non-Uniform)')
plot_quantization_grid(optimized_quantization_levels, title='Quantization Grid (Non-Uniform)')


# Large scale quantization
# vecotor v uniformly between -4 and 4
v = np.random.uniform(-4, 4, 1000)

# Uniform Quantization Grid Setup
num_bits = 3
num_levels = 2 ** num_bits
min_level, max_level = -4.0, 4.0
uniform_quantization_levels = np.linspace(min_level, max_level, num_levels)

# Simple Uniform Quantization
quantized_vector_uniform = np.array([
    uniform_quantization_levels[np.argmin(np.abs(uniform_quantization_levels - x))]
    for x in v
])
l2_norm_uniform = np.linalg.norm(v - quantized_vector_uniform)
print("L2 Norm (Uniform Quantization):", l2_norm_uniform)

# Visualization for uniform quantization
plot_original_vs_quantized(v, quantized_vector_uniform, title='Original vs. Quantized Vector (Uniform) - Large Scale')
plot_with_quantization_levels(v, uniform_quantization_levels, title='Original Vector with Quantization Levels (Uniform) - Large Scale')
plot_distribution(v, uniform_quantization_levels, title='Distribution of Original Vector and Quantization Levels (Uniform) - Large Scale')
plot_quantization_grid(uniform_quantization_levels, title='Quantization Grid (Uniform) - Large Scale')



# Non-Uniform Quantization - Initializing learnable quantization grid
torch_original_vector = torch.tensor(v, dtype=torch.float32, device=device)
quantization_levels_non_uniform = torch.nn.Parameter(torch.rand(num_levels, device=device) * (max_level - min_level) + min_level)
print("Initial Non-Uniform Quantization Levels:", quantization_levels_non_uniform)

# Optimizer - Using Adam
optimizer = torch.optim.Adam([quantization_levels_non_uniform], lr=0.01)

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
uniform_loss = quantization_loss(torch_original_vector, torch.tensor(uniform_quantization_levels, device=device))
print("Loss with Fixed Uniform Quantization Levels:", uniform_loss.item())

# Quantizing using Optimized Non-Uniform Levels
quantized_vector_non_uniform = np.array([
    optimized_quantization_levels[np.argmin(np.abs(optimized_quantization_levels - x))]
    for x in v
])

# Visualizations for Non-Uniform Quantization
plot_original_vs_quantized(v, quantized_vector_non_uniform, title='Original vs. Quantized Vector (Non-Uniform) - Large Scale')
plot_with_quantization_levels(v, optimized_quantization_levels, title='Original Vector with Quantization Levels (Non-Uniform) - Large Scale')
plot_distribution(v, optimized_quantization_levels, title='Distribution of Original Vector and Quantization Levels (Non-Uniform) - Large Scale')
plot_quantization_grid(optimized_quantization_levels, title='Quantization Grid (Non-Uniform) - Large Scale')






