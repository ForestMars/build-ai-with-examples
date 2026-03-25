def analyze_scaling_efficiency(model_size: int, batch_size: int, world_sizes: list):
    """Analyze theoretical scaling efficiency for different world sizes"""
    
    # Assume these rough numbers for modern hardware
    compute_flops = 312e12  # A100 peak FP16 throughput
    network_bandwidth = 600e9  # NVLink bandwidth in bytes/sec
    
    results = {}
    
    for world_size in world_sizes:
        # Compute time (scales inversely with world size)
        forward_flops = estimate_forward_flops(model_size, batch_size // world_size)
        backward_flops = 2 * forward_flops  # Backward is ~2x forward
        total_compute_time = (forward_flops + backward_flops) / compute_flops
        
        # Communication time (gradient AllReduce)
        gradient_bytes = model_size * 4  # FP32 gradients
        
        # Ring AllReduce: 2 * (N-1)/N * data_size / bandwidth
        allreduce_time = 2 * (world_size - 1) / world_size * gradient_bytes / network_bandwidth
        
        # Total step time
        total_time = total_compute_time + allreduce_time
        
        # Efficiency metrics
        ideal_speedup = world_size
        actual_speedup = (total_compute_time + allreduce_time/world_size) / total_time * world_size
        efficiency = actual_speedup / ideal_speedup
        
        results[world_size] = {
            'compute_time': total_compute_time,
            'communication_time': allreduce_time,
            'total_time': total_time,
            'efficiency': efficiency,
            'communication_fraction': allreduce_time / total_time,
        }
        
    return results

# Example analysis
model_size = 70e9  # 70B parameters
batch_size = 256
world_sizes = [1, 2, 4, 8, 16, 32, 64]

scaling_results = analyze_scaling_efficiency(model_size, batch_size, world_sizes)

print("Data Parallelism Scaling Analysis:")
print("GPUs | Efficiency | Comm% | Speedup")
print("-" * 40)

for world_size in world_sizes:
    result = scaling_results[world_size]
    print(f"{world_size:4d} | {result['efficiency']:8.1%} | {result['communication_fraction']:4.1%} | {result['efficiency'] * world_size:6.1f}x")

