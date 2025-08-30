import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

class TensorParallelTransformerLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, world_size=4):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # Tensor parallel multi-head attention
        self.attention = TensorParallelMultiHeadAttention(d_model, n_heads, world_size)
        
        # Tensor parallel feed-forward network
        assert d_ff % world_size == 0
        local_d_ff = d_ff // world_size
        
        self.ff1 = nn.Linear(d_model, local_d_ff)
        self.ff2 = nn.Linear(local_d_ff, d_model)
        self.activation = nn.ReLU()
        
        # Layer norms (replicated across all GPUs)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.ln1(x)
        attn_out, attn_handle = self.attention(x)
        
        # Wait for attention communication to complete
        attn_handle.wait()
        x = residual + attn_out
        
        # Feed-forward with residual connection
        residual = x
        x = self.ln2(x)
        
        # Column parallelism for first FF layer
        ff1_out = self.ff1(x)
        ff1_activated = self.activation(ff1_out)
        
        # Row parallelism for second FF layer
        ff2_out = self.ff2(ff1_activated)
        
        # AllReduce for row parallel layer
        dist.all_reduce(ff2_out, op=dist.ReduceOp.SUM)
        
        return residual + ff2_out

def train_step(rank, world_size):
    """Single training step on one GPU"""
    setup_distributed(rank, world_size)
    
    # Create model
    model = TensorParallelTransformerLayer(d_model=512, n_heads=8, world_size=world_size)
    model = model.cuda(rank)
    
    # Create some dummy data
    batch_size, seq_len, d_model = 4, 128, 512
    x = torch.randn(batch_size, seq_len, d_model, device=f'cuda:{rank}')
    target = torch.randn(batch_size, seq_len, d_model, device=f'cuda:{rank}')
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient synchronization is handled automatically by autograd
    optimizer.step()
    
    print(f"Rank {rank}: Loss = {loss.item():.4f}")
    
    cleanup_distributed()

# To run this: python -m torch.distributed.launch --nproc_per_node=4 your_script.py
if __name__ == "__main__":
    world_size = 4  # Number of GPUs
    mp.spawn(train_step, args=(world_size,), nprocs=world_size, join=True)
