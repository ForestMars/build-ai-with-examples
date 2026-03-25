def calculate_pipeline_efficiency(num_stages: int, num_micro_batches: int) -> dict:
    """Calculate various pipeline efficiency metrics"""
    
    # Theoretical efficiency (ignoring communication)
    theoretical_efficiency = num_micro_batches / (num_micro_batches + num_stages - 1)
    
    # Memory scaling (linear with micro-batches)
    memory_multiplier = num_micro_batches
    
    # Bubble time (time when not all stages are active)
    bubble_steps = num_stages - 1
    total_steps = num_micro_batches + num_stages - 1
    bubble_percentage = (bubble_steps / total_steps) * 100
    
    return {
        'efficiency': theoretical_efficiency,
        'memory_multiplier': memory_multiplier,
        'bubble_percentage': bubble_percentage,
        'throughput_ratio': theoretical_efficiency,
    }

# Find optimal micro-batch count
for num_micro_batches in [4, 8, 16, 32]:
    metrics = calculate_pipeline_efficiency(8, num_micro_batches)
    print(f"Micro-batches: {num_micro_batches}")
    print(f"  Efficiency: {metrics['efficiency']:.1%}")
    print(f"  Memory: {metrics['memory_multiplier']}x")
    print(f"  Bubble time: {metrics['bubble_percentage']:.1f}%")
    print()
