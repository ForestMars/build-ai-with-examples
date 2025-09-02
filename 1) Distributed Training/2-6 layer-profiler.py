import time
from typing import Dict, List

class LayerProfiler:
    """Profile computational cost of individual layers"""
    
    def __init__(self, model: nn.Module, sample_input: torch.Tensor):
        self.model = model
        self.sample_input = sample_input
        self.layer_costs = {}
        
    def profile_layers(self, num_runs: int = 100) -> Dict[int, float]:
        """Profile the computational cost of each layer"""
        
        with torch.no_grad():
            x = self.sample_input
            
            for layer_id, layer in enumerate(self.model.layers):
                # Warmup
                for _ in range(10):
                    _ = layer(x)
                torch.cuda.synchronize()
                
                # Actual timing
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    output = layer(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / num_runs
                self.layer_costs[layer_id] = avg_time
                
                # Use output as input for next layer
                x = output
                
        return self.layer_costs
    
    def optimal_partitioning(self, num_stages: int) -> List[List[int]]:
        """Find optimal layer partitioning based on computational cost"""
        
        if not self.layer_costs:
            self.profile_layers()
            
        # Sort layers by cost (greedy approximation)
        sorted_layers = sorted(self.layer_costs.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # Initialize stages
        stages = [[] for _ in range(num_stages)]
        stage_costs = [0.0] * num_stages
        
        # Greedy assignment to least loaded stage
        for layer_id, cost in sorted_layers:
            min_stage = min(range(num_stages), key=lambda i: stage_costs[i])
            stages[min_stage].append(layer_id)
            stage_costs[min_stage] += cost
            
        # Sort layers within each stage to maintain model order
        for stage in stages:
            stage.sort()
            
        print("Stage costs:", stage_costs)
        print("Load balance ratio:", max(stage_costs) / min(stage_costs))
        
        return stages
