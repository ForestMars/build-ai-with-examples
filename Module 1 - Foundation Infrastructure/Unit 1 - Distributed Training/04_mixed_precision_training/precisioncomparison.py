import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

class PrecisionComparison:
    """Compare FP32 vs Mixed Precision training"""
    
    def __init__(self, model_size=1000000):
        self.model_fp32 = self.create_test_model(model_size)
        self.model_fp16 = self.create_test_model(model_size)
        
        # Mixed precision requires a gradient scaler
        self.scaler = GradScaler()
        
    def create_test_model(self, size):
        """Create a ridiculously large linear layer for testing"""
        return nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 1000),
            nn.LogSoftmax(dim=-1)
        ).cuda()
        
    def memory_usage_comparison(self):
        """Compare memory usage between precisions"""
        
        # FP32 model
        fp32_params = sum(p.numel() * 4 for p in self.model_fp32.parameters())  # 4 bytes per param
        
        # FP16 would use half
        fp16_params = sum(p.numel() * 2 for p in self.model_fp16.parameters())  # 2 bytes per param
        
        print(f"FP32 model: {fp32_params / 1e9:.2f} GB")
        print(f"FP16 model: {fp16_params / 1e9:.2f} GB")
        print(f"Memory savings: {(1 - fp16_params/fp32_params) * 100:.1f}%")
        
        # But wait, there's more! Activations also benefit
        batch_size, seq_len, hidden_dim = 32, 512, 4096
        activation_fp32 = batch_size * seq_len * hidden_dim * 4  # bytes
        activation_fp16 = batch_size * seq_len * hidden_dim * 2  # bytes
        
        print(f"\nActivation memory per layer:")
        print(f"FP32: {activation_fp32 / 1e6:.1f} MB")
        print(f"FP16: {activation_fp16 / 1e6:.1f} MB")

# Quick demo
comparison = PrecisionComparison()
comparison.memory_usage_comparison()