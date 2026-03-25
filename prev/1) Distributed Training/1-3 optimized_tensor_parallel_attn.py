class OptimizedTensorParallelAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, world_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        self.n_heads_per_gpu = n_heads // world_size
        self.d_k = d_model // n_heads
        self.local_d_model = self.n_heads_per_gpu * self.d_k
        
        # Fused QKV projection for better memory efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * self.local_d_model, bias=False)
        self.output_proj = nn.Linear(self.local_d_model, d_model, bias=False)
        
        # Pre-allocate buffers to avoid memory fragmentation
        self.register_buffer('attn_buffer', torch.empty(1, 1, 1, 1))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Fused QKV computation
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * local_d_model]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape with optimal memory access patterns
        q = q.view(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        
        # Resize attention buffer if needed
        if self.attn_buffer.shape != (batch_size, self.n_heads_per_gpu, seq_len, seq_len):
            self.attn_buffer = self.attn_buffer.new_empty(
                batch_size, self.n_heads_per_gpu, seq_len, seq_len
            )
        
        # In-place attention computation to save memory
        torch.matmul(q, k.transpose(-2, -1), out=self.attn_buffer)
        self.attn_buffer.div_(self.d_k ** 0.5)
        torch.softmax(self.attn_buffer, dim=-1, out=self.attn_buffer)
        
        # Attention output
        context = torch.matmul(self.attn_buffer, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection with communication
        local_out = self.output_proj(context)
        
        # Async communication for better overlap
        handle = dist.all_reduce(local_out, op=dist.ReduceOp.SUM, async_op=True)
        
        # You can do other computation here while communication happens
        # handle.wait()  # Uncomment if you need the result immediately
        
        return local_out, handle  # Return handle for async completion

