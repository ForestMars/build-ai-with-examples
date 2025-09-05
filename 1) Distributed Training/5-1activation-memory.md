import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import time

class ActivationMemoryDemo:
    """Demonstrate the activation memory explosion"""
    
    def __init__(self):
        self.memory_stats = []
        
    def measure_activation_memory(self, model, batch_size=16, seq_len=2048):
        """Measure activation memory growth layer by layer"""
        
        x = torch.randn(batch_size, seq_len, model.d_model).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass through each layer, measuring memory
        for i, layer in enumerate(model.layers):
            x = layer(x)
            current_memory = torch.cuda.memory_allocated()
            layer_memory = (current_memory - initial_memory) / 1e6  # MB
            
            self.memory_stats.append({
                'layer': i,
                'memory_mb': layer_memory,
                'activation_shape': x.shape
            })
            
            if i % 10 == 0:
                print(f"Layer {i}: {layer_memory:.1f}MB total, {x.shape}")
                
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nTotal peak memory: {peak_memory:.1f}MB")
        
        return peak_memory

# Simple transformer layer for testing
class TransformerLayer(nn.Module):
    def __init__(self, d_model=4096, nhead=32, dim_feedforward=16384):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention block
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward block
        ff_out = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + ff_out)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers=20, d_model=4096):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

# Demo the memory explosion
model = SimpleTransformer(num_layers=20).cuda()
demo = ActivationMemoryDemo()
peak_memory = demo.measure_activation_memory(model)

