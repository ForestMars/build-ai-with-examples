Tensor Parallelism: Splitting Attention Heads Across GPUs

Part 1 of "Distributed Training from Scratch"

In 1936, Alan Turing proved that any computation could be reduced to a single machine shuttling symbols on an endless tape. In 2024, we've decided that's not nearly fast enough. So we're going to take those computations, chop them up with the precision of a sushi master, and scatter the pieces across multiple GPUs while praying to the NCCL gods that our gradients synchronize properly.

Welcome to tensor parallelism, the art of splitting individual operations across devices rather than entire layers or data batches. Where data parallelism says "let's run the same model on different data," and pipeline parallelism says "let's put different layers on different devices," tensor parallelism boldly declares "let's put parts of the same layer on different devices and see what happens."

What happens, as it turns out, is both beautiful and terrifying. Beautiful because you can train models that would never fit on a single GPU. Terrifying because you're now debugging synchronization bugs across dozens of CUDA streams while your gradients mysteriously vanish into the void.

The Fundamental Challenge: When Models Don't Fit

Let's start with a harsh reality check. A 70B parameter model in FP16 precision requires roughly 140GB of memory just to store the weights. Add gradients, optimizer states, and activations, and you're looking at 400GB+ of memory. The largest single GPU you can buy today (H100 with 80GB HBM) isn't even close.

This is where tensor parallelism becomes not just useful, but essential. Instead of saying "this layer is too big for one GPU," we say "this matrix multiplication is too big for one GPU, so let's split the matrices themselves."

Multi-Head Attention: The Perfect Victim

Multi-head attention is almost criminally well-suited for tensor parallelism. Consider the canonical transformer attention mechanism:

import torch
import torch.nn as nn
import torch.distributed as dist

class NaiveMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # These are the matrices we're going to split
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False) 
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


Now, here's the key insight: those attention heads are embarrassingly parallel. Head 0 doesn't need to know what Head 7 is doing until the very end when we concatenate and project through W_o. This natural structure makes attention layers perfect candidates for tensor parallelism.

The Art of the Split: Column vs Row Parallelism

Tensor parallelism comes in two main flavors, and understanding the difference will save you from mysterious deadlocks at 3 AM.

Column Parallelism: Split the output dimension. Each GPU gets a subset of the columns. Row Parallelism: Split the input dimension. Each GPU gets a subset of the rows.

For multi-head attention, we use column parallelism on the Q, K, V projections (each GPU handles a subset of heads), and row parallelism on the output projection (each GPU contributes to the final result).

Here's how it works in practice:

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


The Communication Tax: When GPUs Talk to Each Other

That all_reduce call is where the rubber meets the road. It's also where most of your performance goes to die if you're not careful.

AllReduce is a collective operation that combines values from all processes. In our case, each GPU has computed a partial result for the output projection, and we need to sum them all together. This happens in two phases:

Reduce-Scatter: Each GPU gets the sum for a subset of elements

AllGather: Each GPU broadcasts its subset to all other GPUs

The naive implementation above has a problem: we're doing an AllReduce on every forward pass. For a model with dozens of attention layers, this becomes a communication nightmare. The solution is to be smarter about when and where we synchronize.

Memory Layout: The Devil in the Details

Here's something the textbooks don't tell you: tensor parallelism is as much about memory layout as it is about computation. Get the layout wrong, and your GPUs will spend more time shuffling data than doing math.

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


The Gradient Flow: Backward Pass Complications

Forward pass is just the beginning. The real complexity emerges during backpropagation, where gradients need to flow backward through the distributed computation graph.

The key insight is that communication operations need to be differentiable. PyTorch's distributed package handles this for you, but understanding what's happening under the hood is crucial for debugging.

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


Putting It All Together: A Complete Example

Let's build a simple but complete tensor parallel transformer layer that you can actually run:

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


Performance Considerations: The Good, The Bad, The Ugly

The Good: Tensor parallelism can give you near-linear speedup for the forward pass when your model is compute-bound and your interconnect is fast (NVLink, InfiniBand).

The Bad: Communication overhead can dominate for smaller models or slower interconnects. Every AllReduce is a synchronization point that can create bubbles in your pipeline.

The Ugly: Memory overhead from storing partial results, complexity of debugging distributed failures, and the fact that your model architecture now depends on your hardware configuration.

Debugging Distributed Nightmares

When (not if) your tensor parallel training hangs or produces NaNs, here's your debugging toolkit:

# Enable detailed distributed logging
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Check for synchronization issues
def debug_sync_point(name):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} reached {name}")
        dist.barrier()  # Force synchronization
        print(f"Rank {dist.get_rank()} passed {name}")

# Monitor gradient norms across ranks
def check_gradient_sync(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Rank {dist.get_rank()}, {name}: grad_norm = {grad_norm}")


The Road Ahead

Tensor parallelism is just the beginning. In the next post, we'll dive into pipeline parallelism, where we'll split models across layers instead of within layers, introducing the delightful complexity of micro-batching and pipeline bubbles.

But for now, you have the tools to split attention heads across GPUs like a distributed computing samurai. The code above will actually run (assuming you have multiple GPUs and your NCCL installation isn't cursed by the networking gods).

The universal Turing machine could simulate any computation on a single tape. We've decided that's not fast enough, so we're simulating our computations across multiple tapes simultaneously. Turing would either be proud or deeply confused. Probably both.

Next up: Pipeline Parallelism - Where we split models by layers and learn to love micro-batching bubbles.

Have questions about tensor parallelism? Want to share your own distributed training war stories? Drop them in the comments. Misery loves company, especially when that misery involves CUDA out-of-memory errors at 3 AM. 
