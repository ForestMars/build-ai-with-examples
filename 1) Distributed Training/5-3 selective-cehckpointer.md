class SelectiveCheckpointer:
    """More sophisticated checkpointing strategy"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.checkpoint_decisions = {}
        
    def analyze_layer_costs(self, sample_input, num_runs=10):
        """Profile computation vs memory cost for each layer"""
        
        layer_stats = {}
        x = sample_input
        
        for i, layer in enumerate(self.model.layers):
            # Measure computation time
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            for _ in range(num_runs):
                _ = layer(x)
            
            torch.cuda.synchronize()
            compute_time = (time.perf_counter() - start_time) / num_runs
            
            # Measure memory usage
            torch.cuda.reset_peak_memory_stats()
            x = layer(x)
            memory_used = torch.cuda.max_memory_allocated() / 1e6
            
            # Store statistics
            layer_stats[i] = {
                'compute_time_ms': compute_time * 1000,
                'memory_mb': memory_used,
                'memory_per_ms': memory_used / (compute_time * 1000)
            }
            
        return layer_stats
        
    def optimize_checkpoint_placement(self, layer_stats, memory_budget_mb=1000):
        """Decide which layers to checkpoint based on cost-benefit"""
        
        # Sort layers by memory/compute ratio (higher = better to checkpoint)
        candidates = [(i, stats['memory_per_ms']) for i, stats in layer_stats.items()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        total_memory_saved = 0
        checkpoint_layers = []
        
        for layer_id, ratio in candidates:
            layer_memory = layer_stats[layer_id]['memory_mb']
            
            if total_memory_saved + layer_memory <= memory_budget_mb:
                checkpoint_layers.append(layer_id)
                total_memory_saved += layer_memory
            else:
                break
                
        print(f"Optimal Checkpointing Strategy:")
        print(f"  Checkpoint layers: {checkpoint_layers}")
        print(f"  Memory saved: {total_memory_saved:.1f}MB")
        print(f"  Additional compute: {sum(layer_stats[i]['compute_time_ms'] for i in checkpoint_layers):.1f}ms")
        
        return checkpoint_layers

class AdaptiveCheckpointWrapper(nn.Module):
    """Wrapper that applies selective checkpointing"""
    
    def __init__(self, model: nn.Module, checkpoint_layers=None):
        super().__init__()
        self.model = model
        self.checkpoint_layers = set(checkpoint_layers or [])
        
    def forward(self, x):
        for i, layer in enumerate(self.model.layers):
            if i in self.checkpoint_layers:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return x

# Example usage
model = SimpleTransformer(num_layers=10, d_model=1024).cuda()
optimizer_tool = SelectiveCheckpointer(model)

# Analyze costs
sample_input = torch.randn(4, 256, 1024).cuda()
layer_stats = optimizer_tool.analyze_layer_costs(sample_input)

# Optimize placement
optimal_checkpoints = optimizer_tool.optimize_checkpoint_placement(layer_stats)

# Apply selective checkpointing
optimized_model = AdaptiveCheckpointWrapper(model, optimal_checkpoints)

