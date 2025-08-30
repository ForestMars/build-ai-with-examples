class TensorParallelLinear(nn.Module):
    """A linear layer split across multiple GPUs with proper gradient handling"""
    
    def __init__(self, in_features, out_features, world_size, dim=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.dim = dim  # Dimension to split: 0 for row, 1 for column
        
        if dim == 1:  # Column parallelism
            assert out_features % world_size == 0
            local_out_features = out_features // world_size
            self.weight = nn.Parameter(torch.randn(local_out_features, in_features))
        else:  # Row parallelism
            assert in_features % world_size == 0
            local_in_features = in_features // world_size
            self.weight = nn.Parameter(torch.randn(out_features, local_in_features))
            
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
    def forward(self, x):
        if self.dim == 1:  # Column parallelism
            # Each GPU computes a slice of the output
            local_output = torch.matmul(x, self.weight.t())
            # Gather results from all GPUs
            output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(output_list, local_output)
            output = torch.cat(output_list, dim=-1)
            
        else:  # Row parallelism
            # Split input across GPUs
            local_x = x[..., self.rank * x.shape[-1] // self.world_size:
                          (self.rank + 1) * x.shape[-1] // self.world_size]
            local_output = torch.matmul(local_x, self.weight.t())
            # Sum results from all GPUs
            output = local_output.clone()
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            
        if self.bias is not None:
            output += self.bias
            
        return output
