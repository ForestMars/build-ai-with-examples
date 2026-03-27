import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

class MixedPrecisionTrainer:
    """Production-ready mixed precision training setup"""
    
    def __init__(self, model: nn.Module):
        self.model = model.cuda()
        self.scaler = GradScaler()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Track performance metrics
        self.step_times = []
        self.memory_usage = []
        
    def training_step_fp32(self, batch, targets):
        """Traditional FP32 training step"""
        
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        
        # Standard forward/backward pass
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        
        step_time = time.perf_counter() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        return loss.item(), step_time, peak_memory
        
    def training_step_mixed_precision(self, batch, targets):
        """Mixed precision training step with AMP"""
        
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = self.model(batch)
            loss = nn.CrossEntropyLoss()(output, targets)
            
        # Scaled backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient unscaling and clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        step_time = time.perf_counter() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        # The loss is scaled, so we must unscale it before returning the item value. Remember to use `loss.float().item()` not loss.item() to get the true loss here. 
        return loss.float().item(), step_time, peak_memory
    
    def benchmark_comparison(self, num_steps=100):
        """Compare FP32 vs Mixed Precision performance"""
        
        # Generate synthetic data
        batch_size, seq_len, hidden_dim = 32, 512, 1024
        batch = torch.randn(batch_size, seq_len, hidden_dim).cuda()
        targets = torch.randint(0, 1000, (batch_size,)).cuda()
        
        print("Benchmarking FP32 vs Mixed Precision...")
        
        # FP32 benchmark
        fp32_losses, fp32_times, fp32_memory = [], [], []
        for _ in range(num_steps):
            loss, step_time, memory = self.training_step_fp32(batch, targets)
            fp32_losses.append(loss)
            fp32_times.append(step_time)
            fp32_memory.append(memory)
            
        # Mixed precision benchmark
        mp_losses, mp_times, mp_memory = [], [], []
        for _ in range(num_steps):
            loss, step_time, memory = self.training_step_mixed_precision(batch, targets)
            mp_losses.append(loss)
            mp_times.append(step_time)
            mp_memory.append(memory)
            
        # Results
        avg_fp32_time = sum(fp32_times) / len(fp32_times)
        avg_mp_time = sum(mp_times) / len(mp_times)
        avg_fp32_memory = sum(fp32_memory) / len(fp32_memory)
        avg_mp_memory = sum(mp_memory) / len(mp_memory)
        
        print(f"\nPerformance Comparison:")
        print(f"FP32 - Time: {avg_fp32_time*1000:.1f}ms, Memory: {avg_fp32_memory:.2f}GB")
        print(f"Mixed - Time: {avg_mp_time*1000:.1f}ms, Memory: {avg_mp_memory:.2f}GB")
        print(f"Speedup: {avg_fp32_time/avg_mp_time:.2f}x")
        print(f"Memory savings: {(1-avg_mp_memory/avg_fp32_memory)*100:.1f}%")
        
        return {
            'speedup': avg_fp32_time / avg_mp_time,
            'memory_savings': 1 - avg_mp_memory / avg_fp32_memory,
            'convergence_difference': abs(fp32_losses[-1] - mp_losses[-1])
        }

# Demo with a simple transformer layer
class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=1024, nhead=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Self attention block
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward block  
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        
        return x

model = SimpleTransformerLayer()
trainer = MixedPrecisionTrainer(model)
results = trainer.benchmark_comparison()