class TensorParallelMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, world_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # Each GPU handles n_heads // world_size heads
        assert n_heads % world_size == 0, "Heads must be divisible by world size"
        self.n_heads_per_gpu = n_heads // world_size
        self.d_k = d_model // n_heads
        
        # Column parallelism: each GPU gets a slice of the output
        local_d_model = d_model // world_size
        self.W_q = nn.Linear(d_model, local_d_model, bias=False)
        self.W_k = nn.Linear(d_model, local_d_model, bias=False)
        self.W_v = nn.Linear(d_model, local_d_model, bias=False)
        
        # Row parallelism: each GPU contributes to the full output
        self.W_o = nn.Linear(local_d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Each GPU computes its subset of heads
        Q = self.W_q(x)  # [batch, seq_len, local_d_model]
        K = self.W_k(x)  # [batch, seq_len, local_d_model] 
        V = self.W_v(x)  # [batch, seq_len, local_d_model]
        
        # Reshape for our local heads
        Q = Q.reshape(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads_per_gpu, self.d_k).transpose(1, 2)
        
        # Local attention computation (no communication needed!)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Local output projection
        local_output = self.W_o(context)
        
        # HERE'S THE MAGIC: AllReduce to combine results from all GPUs
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        
        return local_output

