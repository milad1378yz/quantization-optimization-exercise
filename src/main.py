import numpy as np
import torch
import argparse
import os
import time
import multiprocessing
import logging
from functools import partial
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


def setup_logger(save_dir):
    """Set up the logger to output to console and file."""
    logger = logging.getLogger("QuantizationLogger")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(save_dir, "logs.txt"), mode="w")
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters
    c_format = logging.Formatter("%(message)s")
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Add formatters to handlers
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


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
        logger.debug(f"Using a sample of size {sample_size} for optimization.")
    else:
        vector_sample = vector

    torch_vector = torch.tensor(vector_sample, dtype=torch.float32, device=device)
    q_levels = torch.nn.Parameter(
        torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
    )
    optimizer = torch.optim.Adam([q_levels], lr=0.01)

    logger.debug("Starting optimization of quantization levels.")
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = quantization_loss(torch_vector, q_levels)
        loss.backward()
        optimizer.step()

        # Log loss every 50 epochs
        if epoch % 50 == 0 or epoch == 1 or epoch == num_epochs:
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")

    optimized_levels = q_levels.detach().cpu().numpy()
    logger.debug("Optimization of quantization levels completed.")

    # Quantize the full vector using the optimized levels
    quantized_vector = quantize_vector_in_batches(
        vector, optimized_levels, use_multiprocessing=args.use_multiprocessing
    )

    l2_norm = np.linalg.norm(vector - quantized_vector)
    return quantized_vector, optimized_levels, l2_norm


def quantize_batch(batch, levels):
    """Helper function to quantize a single batch."""
    return levels[np.argmin(np.abs(batch[:, None] - levels[None, :]), axis=1)]


def quantize_vector_in_batches(vector, levels, batch_size=1000000, use_multiprocessing=False):
    """Quantize a large vector in batches to handle memory constraints."""
    quantized_vector = np.empty_like(vector)
    num_batches = int(np.ceil(len(vector) / batch_size))
    logger.debug(f"Quantizing vector in {num_batches} batches.")

    if use_multiprocessing and num_batches > 1:
        logger.debug("Using multiprocessing for batch quantization.")
        # Prepare batches
        batches = [
            vector[i * batch_size : min((i + 1) * batch_size, len(vector))]
            for i in range(num_batches)
        ]
        with multiprocessing.Pool() as pool:
            results = pool.map(partial(quantize_batch, levels=levels), batches)
        # Combine results
        for i, quantized_batch in enumerate(results):
            start_idx = i * batch_size
            end_idx = start_idx + len(quantized_batch)
            quantized_vector[start_idx:end_idx] = quantized_batch
    else:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(vector))
            batch = vector[start_idx:end_idx]
            quantized_batch = quantize_batch(batch, levels)
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


def quantize_vector_with_outliers_in_batches(
    vector, levels, threshold=3.0, batch_size=1000000, use_multiprocessing=False
):
    """Quantize a large vector with outlier handling in batches."""
    quantized_vector = np.empty_like(vector)
    num_batches = int(np.ceil(len(vector) / batch_size))
    logger.debug(f"Quantizing vector with outliers in {num_batches} batches.")

    mean = np.mean(vector)
    std = np.std(vector)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    if use_multiprocessing and num_batches > 1:
        logger.debug("Using multiprocessing for batch quantization with outliers.")
        # Prepare batches
        batches = [
            vector[i * batch_size : min((i + 1) * batch_size, len(vector))]
            for i in range(num_batches)
        ]
        # Prepare partial function with fixed parameters
        quantize_partial = partial(
            quantize_batch_with_outliers,
            levels=levels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        with multiprocessing.Pool() as pool:
            results = pool.map(quantize_partial, batches)
        # Combine results
        for i, quantized_batch in enumerate(results):
            start_idx = i * batch_size
            end_idx = start_idx + len(quantized_batch)
            quantized_vector[start_idx:end_idx] = quantized_batch
    else:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(vector))
            batch = vector[start_idx:end_idx]
            quantized_batch = quantize_batch_with_outliers(batch, levels, lower_bound, upper_bound)
            quantized_vector[start_idx:end_idx] = quantized_batch

    return quantized_vector


def quantize_batch_with_outliers(batch, levels, lower_bound, upper_bound):
    """Helper function to quantize a single batch with outlier handling."""
    outlier_mask = (batch < lower_bound) | (batch > upper_bound)
    quantized_batch = np.where(
        outlier_mask,
        batch,  # Keep outliers in full precision
        levels[np.argmin(np.abs(batch[:, None] - levels[None, :]), axis=1)],
    )
    return quantized_batch


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
    Similar to non_uniform_quantization, but we exclude outliers from the optimization:
    L(q) = (1/N_normal) * Σ_i [min_k (v_i - q_k)^2], for i where v_i is not an outlier
    """
    num_levels = 2**num_bits

    # Use sampling for large vectors to accelerate optimization
    if sample_size is not None and len(vector) > sample_size:
        sample_indices = np.random.choice(len(vector), sample_size, replace=False)
        vector_sample = vector[sample_indices]
        logger.debug(f"Using a sample of size {sample_size} for optimization.")
    else:
        vector_sample = vector

    torch_vector = torch.tensor(vector_sample, dtype=torch.float32, device=device)
    outlier_mask = identify_outliers(torch_vector, threshold)
    q_levels = torch.nn.Parameter(
        torch.rand(num_levels, device=device) * (max_level - min_level) + min_level
    )
    optimizer = torch.optim.Adam([q_levels], lr=0.01)

    logger.debug("Starting optimization of quantization levels with outlier handling.")
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = quantization_loss_with_outliers(torch_vector, q_levels, outlier_mask)
        loss.backward()
        optimizer.step()

        # Log loss every 50 epochs
        if epoch % 50 == 0 or epoch == 1 or epoch == num_epochs:
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")

    optimized_levels = q_levels.detach().cpu().numpy()
    logger.debug("Optimization of quantization levels with outlier handling completed.")

    # Quantize the full vector in batches
    quantized_vector = quantize_vector_with_outliers_in_batches(
        vector, optimized_levels, threshold=threshold, use_multiprocessing=args.use_multiprocessing
    )

    l2_norm = np.linalg.norm(vector - quantized_vector)
    return quantized_vector, optimized_levels, l2_norm


def plot_results(vector, quantized_vector, levels, approach_name, save_dir):
    """Generate and save plots for quantization results."""
    # For large vectors, skip plotting due to performance constraints
    if len(vector) > 100000:
        logger.info(f"Skipping plots for large vector in {approach_name}.")
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
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save plots and logs"
    )
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
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Enable multiprocessing for batch processing",
    )
    args = parser.parse_args()
    return args


def main():
    global args  # Make args global to access in functions
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    global logger  # Make logger global to access in functions
    logger = setup_logger(args.save_dir)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Generating vector of size: {args.vector_size}")
    vector = generate_vector(size=args.vector_size, seed=args.seed)
    logger.info(f"Vector generated with {len(vector)} elements.")

    approaches = (
        [args.approach] if args.approach != "all" else ["uniform", "non-uniform", "easyquant"]
    )

    results = {}
    for approach in approaches:
        logger.info(f"\nStarting {approach.title()} Quantization...")
        start_time = time.time()  # Start timing
        if approach == "uniform":
            quantized_vector, levels, l2_norm = uniform_quantization(vector, num_bits=args.num_bits)
            latency = time.time() - start_time  # Compute latency
            logger.info(f"L2 Norm ({approach.title()} Quantization): {l2_norm}")
            logger.info(f"Latency ({approach.title()} Quantization): {latency:.4f} seconds")
            plot_results(vector, quantized_vector, levels, approach.title(), args.save_dir)
            results[approach] = {"l2_norm": l2_norm, "latency": latency}
        elif approach == "non-uniform":
            quantized_vector, levels, l2_norm = non_uniform_quantization(
                vector,
                num_bits=args.num_bits,
                num_epochs=500,
                device=device,
                sample_size=args.sample_size,
            )
            latency = time.time() - start_time  # Compute latency
            logger.info(f"L2 Norm ({approach.title()} Quantization): {l2_norm}")
            logger.info(f"Latency ({approach.title()} Quantization): {latency:.4f} seconds")
            plot_results(vector, quantized_vector, levels, approach.title(), args.save_dir)
            results[approach] = {"l2_norm": l2_norm, "latency": latency}
        elif approach == "easyquant":
            quantized_vector, levels, l2_norm = non_uniform_quantization_with_outliers(
                vector,
                num_bits=args.num_bits,
                num_epochs=500,
                device=device,
                threshold=args.threshold,
                sample_size=args.sample_size,
            )
            latency = time.time() - start_time  # Compute latency
            logger.info(f"L2 Norm (EasyQuant Quantization): {l2_norm}")
            logger.info(f"Latency (EasyQuant Quantization): {latency:.4f} seconds")
            plot_results(vector, quantized_vector, levels, "EasyQuant", args.save_dir)
            results[approach] = {"l2_norm": l2_norm, "latency": latency}
        else:
            logger.error(f"Invalid approach selected: {approach}")
            raise ValueError(f"Invalid approach selected: {approach}")

    # Log final results
    logger.info("\nFinal Results:")
    for approach, metrics in results.items():
        logger.info(
            f"{approach.title()} Quantization - L2 Norm: {metrics['l2_norm']}, Latency: {metrics['latency']:.4f} seconds"
        )

    # Save results to 'results.txt' in save_dir
    results_path = os.path.join(args.save_dir, "results.txt")
    with open(results_path, "w") as f:
        for approach, metrics in results.items():
            f.write(f"{approach.title()} Quantization:\n")
            f.write(f"L2 Norm: {metrics['l2_norm']}\n")
            f.write(f"Latency: {metrics['latency']:.4f} seconds\n\n")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
