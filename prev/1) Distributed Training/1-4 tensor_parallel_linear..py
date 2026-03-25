import torch
import torch.nn as nn
import torch.distributed as dist

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
            
            # Register gradient hook for weight synchronization
            self.weight.register_hook(self._all_reduce_grad_hook)
            if self.bias is not None:
                self.bias.register_hook(self._all_reduce_grad_hook)
        
        # Row parallelism (sharding the input features)
        else:
            assert in_features % world_size == 0, "in_features must be divisible by world_size for row parallelism"
            local_in_features = in_features // world_size
            self.weight = nn.Parameter(torch.randn(local_in_features, out_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            
            # Register gradient hook for weight synchronization
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
            # Each GPU computes a slice of the output.
            # The input x is replicated across all GPUs.
            local_output = torch.matmul(x, self.weight)
            
            # Add bias before gather (bias is already sharded correctly)
            if self.bias_enabled:
                local_output += self.bias
            
            # all_gather operation ensures output is avail on all ranks.
            output = self.gather_op(local_output)
            
        else:  # Row Parallelism
            # The input x is already split across GPUs.
            # No manual slicing. Each GPU receives a local slice.
            local_output = torch.matmul(x, self.weight)
            
            # all_reduce operation sums partial results from each GPU.
            output = self.reduce_op(local_output)
            
            if self.bias_enabled:
                # The bias is not sharded, so it's applied normally.
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