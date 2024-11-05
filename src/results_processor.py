import re
import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = 'results/results.txt'

# Initialize an empty list to collect data rows
data = []

# Regex patterns for extracting data
vector_size_pattern = re.compile(r"Vector Size: (\w+)___multiprocess: (True|False)__device: (\w+)")
approach_pattern = re.compile(r"^(.*) Quantization:")
l2_norm_pattern = re.compile(r"L2 Norm: ([\d.e-]+)")
latency_pattern = re.compile(r"Latency: ([\d.]+) seconds")

# Open and read the file
with open(file_path, 'r') as file:
    vector_size, multiprocessing, device = None, None, None
    approach, l2_norm, latency = None, None, None
    
    for line in file:
        line = line.strip()
        
        # Match vector size, multiprocessing, and device
        vector_match = vector_size_pattern.match(line)
        if vector_match:
            vector_size, multiprocessing, device = vector_match.groups()
            continue
        
        # Match approach
        approach_match = approach_pattern.match(line)
        if approach_match:
            approach = approach_match.group(1)
            continue
        
        # Match L2 Norm
        l2_norm_match = l2_norm_pattern.match(line)
        if l2_norm_match:
            l2_norm = float(l2_norm_match.group(1))
            continue
        
        # Match Latency
        latency_match = latency_pattern.match(line)
        if latency_match:
            latency = float(latency_match.group(1))
            # Append row to data after finding all details for a row
            data.append([approach, vector_size, l2_norm, latency, multiprocessing, device])
            # Reset approach, l2_norm, and latency for the next block
            approach, l2_norm, latency = None, None, None

# Create DataFrame from parsed data
df = pd.DataFrame(data, columns=['Approach', 'Vector Size', 'L2 Norm', 'Latency (s)', 'Multiprocessing', 'Device'])

# Separate CPU and GPU data
df_cpu = df[df['Device'] == 'cpu']
df_gpu = df[df['Device'] == 'cuda']

# Further separate small and large vector data for CPU and GPU
df_cpu_small = df_cpu[df_cpu['Vector Size'] == 'small']
df_cpu_large = df_cpu[df_cpu['Vector Size'] == 'large']
df_gpu_small = df_gpu[df_gpu['Vector Size'] == 'small']
df_gpu_large = df_gpu[df_gpu['Vector Size'] == 'large']

# Plotting with separate subplots for small and large vectors for both CPU and GPU
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Quantization Performance Comparison on CPU and GPU by Vector Size', fontsize=16)

# CPU - Small Vector - L2 Norm
df_cpu_small.pivot(index='Approach', columns='Multiprocessing', values='L2 Norm').plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('CPU - Small Vector - L2 Norm')
axes[0, 0].set_ylabel('L2 Norm')

# CPU - Small Vector - Latency
df_cpu_small.pivot(index='Approach', columns='Multiprocessing', values='Latency (s)').plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('CPU - Small Vector - Latency (s)')
axes[1, 0].set_ylabel('Latency (s)')

# CPU - Large Vector - L2 Norm
df_cpu_large.pivot(index='Approach', columns='Multiprocessing', values='L2 Norm').plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('CPU - Large Vector - L2 Norm')
axes[0, 1].set_ylabel('L2 Norm')

# CPU - Large Vector - Latency
df_cpu_large.pivot(index='Approach', columns='Multiprocessing', values='Latency (s)').plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('CPU - Large Vector - Latency (s)')
axes[1, 1].set_ylabel('Latency (s)')

# GPU - Small Vector - L2 Norm
df_gpu_small.pivot(index='Approach', columns='Multiprocessing', values='L2 Norm').plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('GPU - Small Vector - L2 Norm')
axes[0, 2].set_ylabel('L2 Norm')

# GPU - Small Vector - Latency
df_gpu_small.pivot(index='Approach', columns='Multiprocessing', values='Latency (s)').plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('GPU - Small Vector - Latency (s)')
axes[1, 2].set_ylabel('Latency (s)')

# GPU - Large Vector - L2 Norm
df_gpu_large.pivot(index='Approach', columns='Multiprocessing', values='L2 Norm').plot(kind='bar', ax=axes[0, 3])
axes[0, 3].set_title('GPU - Large Vector - L2 Norm')
axes[0, 3].set_ylabel('L2 Norm')

# GPU - Large Vector - Latency
df_gpu_large.pivot(index='Approach', columns='Multiprocessing', values='Latency (s)').plot(kind='bar', ax=axes[1, 3])
axes[1, 3].set_title('GPU - Large Vector - Latency (s)')
axes[1, 3].set_ylabel('Latency (s)')

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/quantization_performance_comparison.png')
