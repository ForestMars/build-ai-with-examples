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

class TensorParallelLinear(nn.Module):
    """A linear layer split across multiple GPUs with proper gradient handling."""
    
    def __init__(self, in_features, out_features, world_size, rank, dim=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.dim = dim
        self.bias_enabled = bias

        # Column parallelism (sharding the output features)
        if dim == 1:
            assert out_features % world_size == 0, "out_features must be divisible by world_size for column parallelism"
            local_out_features = out_features // world_size
            self.weight = nn.Parameter(torch.randn(in_features, local_out_features))
            self.bias = nn.Parameter(torch.zeros(local_out_features)) if bias else None
        
        # Row parallelism (sharding the input features)
        else:
            assert in_features % world_size == 0, "in_features must be divisible by world_size for row parallelism"
            local_in_features = in_features // world_size
            self.weight = nn.Parameter(torch.randn(local_in_features, out_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Register gradient hooks for parameter synchronization
        self.weight.register_hook(self._all_reduce_grad_hook)
        if self.bias is not None:
            self.bias.register_hook(self._all_reduce_grad_hook)
    
    def _all_reduce_grad_hook(self, grad):
        """Hook to synchronize gradients across all GPUs."""
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.data.div_(self.world_size)  # Average the gradients in-place
        return grad
        
    def forward(self, x):
        if self.dim == 1:  # Column Parallelism
            # Each GPU computes a slice of the output
            local_output = torch.matmul(x, self.weight)
            
            # Add bias before gather (bias is already sharded correctly)
            if self.bias_enabled:
                local_output += self.bias
            
            # all_gather operation ensures output is available on all ranks
            output = self.gather_op(local_output)
            
        else:  # Row Parallelism
            # The input x is already split across GPUs
            local_output = torch.matmul(x, self.weight)
            
            # all_reduce operation sums partial results from each GPU
            output = self.reduce_op(local_output)
            
            if self.bias_enabled:
                # The bias is not sharded, so it's applied normally
                output += self.bias
                
        return output
    
    # Helper functions to abstract the distributed operations
    def gather_op(self, output):
        output_list = [torch.zeros_like(output) for _ in range(self.world_size)]
        dist.all_gather(output_list, output)
        return torch.cat(output_list, dim=-1)

    def reduce_op(self, output):
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

class TensorParallelTransformerLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, world_size=4):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # Tensor parallel feed-forward network
        # ff1: column parallel (expand), ff2: row parallel (contract)
        self.ff1 = TensorParallelLinear(d_model, d_ff, world_size, self.rank, dim=1, bias=True)
        self.ff2 = TensorParallelLinear(d_ff, d_model, world_size, self.rank, dim=0, bias=True)
        self.activation = nn.ReLU()
        
        # Layer norms (replicated across all GPUs)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Feed-forward with residual connection
        residual = x
        x = self.ln2(x)
        
        # Column parallelism for first FF layer (already handles communication)
        ff1_out = self.ff1(x)
        ff1_activated = self.activation(ff1_out)
        
        # Row parallelism for second FF layer (already handles communication)
        ff2_out = self.ff2(ff1_activated)
        
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
    
    # Gradient synchronization is handled automatically by hooks
    optimizer.step()
    
    print(f"Rank {rank}: Loss = {loss.item():.4f}")
    
    cleanup_distributed()

# To run this: python -m torch.distributed.launch --nproc_per_node=4 your_script.py
if __name__ == "__main__":
    world_size = 4  # Number of GPUs
    mp.spawn(train_step, args=(world_size,), nprocs=world_size, join=True)