import torch
import torch.nn as nn
from typing import List, Optional

class PipelineStage(nn.Module):
    """A single stage in our pipeline (multiple transformer layers)"""
    
    def __init__(self, layers: List[nn.Module], stage_id: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.stage_id = stage_id
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NaivePipelineParallelTransformer(nn.Module):
    """The simplest possible pipeline parallel transformer"""
    
    def __init__(self, total_layers=80, num_stages=8, d_model=512):
        super().__init__()
        self.num_stages = num_stages
        self.layers_per_stage = total_layers // num_stages
        
        # Create all transformer layers
        all_layers = [TransformerLayer(d_model) for _ in range(total_layers)]
        
        # Split layers across stages
        self.stages = nn.ModuleList([
            PipelineStage(
                all_layers[i * self.layers_per_stage:(i + 1) * self.layers_per_stage],
                stage_id=i
            ) for i in range(num_stages)
        ])
        
    def forward(self, x):
        # Sequential execution (this is wrong, but let's start here)
        for stage in self.stages:
            x = stage(x)
        return x