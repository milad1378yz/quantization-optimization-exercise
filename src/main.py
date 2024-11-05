import numpy as np
import torch
import argparse
import os
from visualizer import (
    plot_original_vs_quantized,
    plot_with_quantization_levels,
    plot_distribution,
    plot_quantization_grid,
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_vector(size="small", seed=42):
    """Generate vector based on specified size."""
    if size == "small":
        return np.array([3.2, -1.4, 2.5, -0.9, 1.8, -3.7, 0.0, 4.0, 2.2, -1.3])
    elif size == "large":
        np.random.seed(seed)
        # For practical purposes in this code, we'll use a smaller size
        # but the method will be scalable to larger sizes
        return np.random.normal(0, 1, 10000000)  # 10 million elements as a stand-in
    else:
        raise ValueError("Invalid vector size. Choose 'small' or 'large'.")


def quantization_loss(v, q_levels):
    """
    Compute average distance to closest quantization level.

    Optimization Problem:
    Minimize the L2 norm (mean squared error) between the original vector and the quantized vector.
    Mathematically, we aim to minimize:
    L(q) = (1/N) * Σ_i [min_k (v_i - q_k)^2]
    where:
    - v_i: original vector elements
    - q_k: quantization levels
    - N: number of elements in vector v
    """
    distances = (v.view(-1, 1) - q_levels.view(1, -1)) ** 2
    min_distances, _ = distances.min(dim=1)
    return min_distances.mean()


def uniform_quantization(vector, num_bits=3, min_level=-4.0, max_level=4.0):
    """
    Perform simple uniform quantization.

    Optimization Problem:
    Replace each element of vector v with the nearest quantization level from a uniform grid.
    This minimizes the L2 norm between v and the quantized vector q(v):
    q(v_i) = argmin_l |v_i - l|
    """
    num_levels = 2**num_bits
    levels = np.linspace(min_level, max_level, num_levels)
    quantized_vector = levels[np.argmin(np.abs(vector[:, None] - levels[None, :]), axis=1)]
    l2_norm = np.linalg.norm(vector - quantized_vector)
    return quantized_vector, levels, l2_norm


def non_uniform_quantization(
    vector,
    num_bits=3,
    min_level=-4.0,
    max_level=4.0,
    num_epochs=500,
    device="cpu",
    sample_size=None,
):
    """
    Optimize quantization levels for non-uniform quantization.

    Optimization Problem:
    Find quantization levels q_k that minimize:
    L(q) = (1/N) * Σ_i [min_k (v_i - q_k)^2]

    For large vectors, we use a sample of the data to optimize q_k.
    """
    num_levels = 2**num_bits

    # Use sampling for large vectors to accelerate optimization
    if sample_size is not None and len(vector) > sample_size:
        sample_indices = np.random.choice(len(vector), sample_size, replace=False)
        vector_sample = vector[sample_indices]
    else:
        vector_sample = vector

    torch_vector = torch.tensor(vector_sample, dtype=torch.float32, device=device)
    q_levels = torch.nn.Parameter(
        torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
    )
    optimizer = torch.optim.Adam([q_levels], lr=0.01)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = quantization_loss(torch_vector, q_levels)
        loss.backward()
        optimizer.step()

    optimized_levels = q_levels.detach().cpu().numpy()

    # Quantize the full vector using the optimized levels
    quantized_vector = quantize_vector_in_batches(vector, optimized_levels)

    l2_norm = np.linalg.norm(vector - quantized_vector)
    return quantized_vector, optimized_levels, l2_norm


def quantize_vector_in_batches(vector, levels, batch_size=1000000):
    """Quantize a large vector in batches to handle memory constraints."""
    quantized_vector = np.empty_like(vector)
    num_batches = int(np.ceil(len(vector) / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(vector))
        batch = vector[start_idx:end_idx]
        quantized_batch = levels[np.argmin(np.abs(batch[:, None] - levels[None, :]), axis=1)]
        quantized_vector[start_idx:end_idx] = quantized_batch

    return quantized_vector


def identify_outliers(v, threshold=3.0):
    """Identify outliers in the vector based on a threshold."""
    mean, std = torch.mean(v), torch.std(v)
    outliers = torch.abs(v - mean) > (threshold * std)
    return outliers


def quantization_loss_with_outliers(v, q_levels, mask):
    """
    Compute quantization loss, ignoring outliers based on mask.

    Optimization Problem:
    Minimize the L2 norm between the quantized vector and the original vector,
    excluding outliers:
    L(q) = (1/N_normal) * Σ_i [min_k (v_i - q_k)^2], for i where mask_i is False
    """
    v_normal = v[~mask]
    distances = (v_normal.view(-1, 1) - q_levels.view(1, -1)) ** 2
    min_distances, _ = distances.min(dim=1)
    return min_distances.mean()


def non_uniform_quantization_with_outliers(
    vector,
    num_bits=3,
    min_level=-4.0,
    max_level=4.0,
    num_epochs=500,
    device="cpu",
    threshold=3.0,
    sample_size=None,
):
    """
    Optimize quantization levels, isolating outliers.

    Optimization Problem:
    Similar to non-uniform quantization, but we exclude outliers from the optimization:
    L(q) = (1/N_normal) * Σ_i [min_k (v_i - q_k)^2], for i where v_i is not an outlier
    """
    num_levels = 2**num_bits

    # Use sampling for large vectors to accelerate optimization
    if sample_size is not None and len(vector) > sample_size:
        sample_indices = np.random.choice(len(vector), sample_size, replace=False)
        vector_sample = vector[sample_indices]
    else:
        vector_sample = vector

    torch_vector = torch.tensor(vector_sample, dtype=torch.float32, device=device)
    outlier_mask = identify_outliers(torch_vector, threshold)
    q_levels = torch.nn.Parameter(
        torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
    )
    optimizer = torch.optim.Adam([q_levels], lr=0.01)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = quantization_loss_with_outliers(torch_vector, q_levels, outlier_mask)
        loss.backward()
        optimizer.step()

    optimized_levels = q_levels.detach().cpu().numpy()

    # Quantize the full vector in batches
    quantized_vector = quantize_vector_with_outliers_in_batches(vector, optimized_levels, threshold)

    l2_norm = np.linalg.norm(vector - quantized_vector)
    return quantized_vector, optimized_levels, l2_norm


def quantize_vector_with_outliers_in_batches(vector, levels, threshold=3.0, batch_size=1000000):
    """Quantize a large vector with outlier handling in batches."""
    quantized_vector = np.empty_like(vector)
    num_batches = int(np.ceil(len(vector) / batch_size))

    mean = np.mean(vector)
    std = np.std(vector)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(vector))
        batch = vector[start_idx:end_idx]
        outlier_mask = (batch < lower_bound) | (batch > upper_bound)
        quantized_batch = np.where(
            outlier_mask,
            batch,  # Keep outliers in full precision
            levels[np.argmin(np.abs(batch[:, None] - levels[None, :]), axis=1)],
        )
        quantized_vector[start_idx:end_idx] = quantized_batch

    return quantized_vector


def plot_results(vector, quantized_vector, levels, approach_name, save_dir):
    """Generate and save plots for quantization results."""
    # For large vectors, skip plotting due to performance constraints
    if len(vector) > 100000:
        print(f"Skipping plots for large vector in {approach_name}.")
        return

    plot_original_vs_quantized(
        vector,
        quantized_vector,
        title=f"Original vs. Quantized Vector ({approach_name})",
        save_path=os.path.join(
            save_dir, f'original_vs_quantized_{approach_name.lower().replace(" ", "_")}.png'
        ),
    )
    plot_with_quantization_levels(
        vector,
        levels,
        title=f"Original Vector with Quantization Levels ({approach_name})",
        save_path=os.path.join(
            save_dir, f'vector_with_levels_{approach_name.lower().replace(" ", "_")}.png'
        ),
    )
    plot_distribution(
        vector,
        levels,
        title=f"Distribution of Original Vector and Quantization Levels ({approach_name})",
        save_path=os.path.join(
            save_dir, f'distribution_{approach_name.lower().replace(" ", "_")}.png'
        ),
    )
    plot_quantization_grid(
        levels,
        title=f"Quantization Grid ({approach_name})",
        save_path=os.path.join(
            save_dir, f'quantization_grid_{approach_name.lower().replace(" ", "_")}.png'
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Quantization Techniques")
    parser.add_argument(
        "--vector_size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Size of the vector to quantize (small or large)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument(
        "--approach",
        type=str,
        default="all",
        choices=["uniform", "non-uniform", "easyquant", "all"],
        help="Quantization approach to use",
    )
    parser.add_argument("--num_bits", type=int, default=3, help="Number of bits for quantization")
    parser.add_argument(
        "--threshold", type=float, default=3.0, help="Threshold for outlier detection in EasyQuant"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000000,
        help="Sample size for optimizing quantization levels in large vectors",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vector = generate_vector(size=args.vector_size, seed=args.seed)

    approaches = (
        [args.approach] if args.approach != "all" else ["uniform", "non-uniform", "easyquant"]
    )

    results = {}
    for approach in approaches:
        if approach == "uniform":
            quantized_vector, levels, l2_norm = uniform_quantization(vector, num_bits=args.num_bits)
            print(f"L2 Norm ({approach.title()} Quantization): {l2_norm}")
            plot_results(vector, quantized_vector, levels, approach.title(), args.save_dir)
            results[approach] = {"l2_norm": l2_norm}
        elif approach == "non-uniform":
            quantized_vector, levels, l2_norm = non_uniform_quantization(
                vector,
                num_bits=args.num_bits,
                num_epochs=500,
                device=device,
                sample_size=args.sample_size,
            )
            print(f"L2 Norm ({approach.title()} Quantization): {l2_norm}")
            plot_results(vector, quantized_vector, levels, approach.title(), args.save_dir)
            results[approach] = {"l2_norm": l2_norm}
        elif approach == "easyquant":
            quantized_vector, levels, l2_norm = non_uniform_quantization_with_outliers(
                vector,
                num_bits=args.num_bits,
                num_epochs=500,
                device=device,
                threshold=args.threshold,
                sample_size=args.sample_size,
            )
            print(f"L2 Norm (EasyQuant Quantization): {l2_norm}")
            plot_results(vector, quantized_vector, levels, "EasyQuant", args.save_dir)
            results[approach] = {"l2_norm": l2_norm}
        else:
            raise ValueError(f"Invalid approach selected: {approach}")

    print(results)


if __name__ == "__main__":
    main()
