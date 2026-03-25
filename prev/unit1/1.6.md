# ZeRO Optimizer States: Sharding Optimizer Memory

**Part 6 of "Distributed Training from Scratch"**

In 1965, Gordon Moore observed that the number of transistors on a chip doubles every two years. In 2019, Microsoft Research observed that optimizer states in deep learning triple the memory footprint of your model, and somebody should probably do something about that. Enter ZeRO (Zero Redundancy Optimizer), the realization that storing identical copies of Adam's momentum and variance tensors across every GPU is not just wasteful—it's almost criminally inefficient.

ZeRO represents a fundamental shift in how we think about distributed training. Instead of "every GPU has everything," we embrace "every GPU has something unique." It's the computational equivalent of realizing that your entire team doesn't need to carry identical copies of the same 500-page manual—you can tear it up, give everyone a section, and share when needed.

## The Optimizer Memory Problem

Let's do some uncomfortable math. A 7B parameter model requires 28GB just for weights in FP32. But Adam optimizer? That needs momentum (28GB) plus variance (28GB) plus gradients (28GB) plus the weights themselves. You're looking at 112GB for a model that started at 28GB. The optimizer state is literally 4x larger than the model.

Traditional data parallelism says: "No problem, put identical copies on every GPU!" Which works until you realize you're spending $300K on hardware to store the same momentum tensors eight times over.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple
import numpy as np

class OptimizerMemoryAnalyzer:
    """Analyze optimizer memory overhead"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_count = sum(p.numel() for p in model.parameters())
        
    def calculate_memory_breakdown(self):
        """Calculate memory usage for different optimizers"""
        
        param_memory = self.param_count * 4  # FP32 weights
        gradient_memory = self.param_count * 4  # FP32 gradients
        
        optimizers = {
            'SGD': {
                'momentum': param_memory,  # momentum buffer
                'total_multiplier': 3  # weights + gradients + momentum
            },
            'Adam': {
                'momentum': param_memory,  # first moment
                'variance': param_memory,  # second moment  
                'total_multiplier': 4  # weights + grads + m + v
            },
            'AdamW': {
                'momentum': param_memory,
                'variance': param_memory,
                'total_multiplier': 4
            }
        }
        
        print(f"Model Analysis ({self.param_count/1e6:.1f}M parameters):")
        print(f"  Parameter memory: {param_memory/1e9:.2f}GB")
        print(f"  Gradient memory: {gradient_memory/1e9:.2f}GB")
        print()
        
        for name, config in optimizers.items():
            total_memory = param_memory * config['total_multiplier']
            print(f"{name} Optimizer:")
            print(f"  Total memory: {total_memory/1e9:.2f}GB")
            print(f"  Multiplier: {config['total_multiplier']}x model size")
            print()
            
        return optimizers

class TraditionalDataParallelOverhead:
    """Demonstrate the memory waste in traditional data parallelism"""
    
    def __init__(self, model_size_gb: float, world_size: int):
        self.model_size_gb = model_size_gb
        self.world_size = world_size
        
    def calculate_redundancy(self):
        """Calculate memory redundancy across GPUs"""
        
        # Traditional DP: every GPU stores everything
        total_model_memory = self.model_size_gb * 4  # Adam multiplier
        total_cluster_memory = total_model_memory * self.world_size
        
        # Actual unique data
        unique_data = self.model_size_gb + (self.model_size_gb * self.world_size)  # weights + distributed gradients
        
        redundancy = total_cluster_memory / unique_data
        wasted_memory = total_cluster_memory - unique_data
        
        print(f"Traditional Data Parallel Memory Analysis:")
        print(f"  Model size: {self.model_size_gb}GB")
        print(f"  Per-GPU memory (Adam): {total_model_memory}GB")
        print(f"  Total cluster memory: {total_cluster_memory}GB")
        print(f"  Unique data: {unique_data}GB")
        print(f"  Redundancy factor: {redundancy:.1f}x")
        print(f"  Wasted memory: {wasted_memory}GB")
        
        return redundancy

# Demo the problem
class SimpleModel(nn.Module):
    def __init__(self, size=7e9):  # 7B parameters
        super().__init__()
        # Simplified representation of a large model
        hidden_dim = int(np.sqrt(size))
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1000)  # classifier
        
    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

model = SimpleModel(7e9)
analyzer = OptimizerMemoryAnalyzer(model)
memory_breakdown = analyzer.calculate_memory_breakdown()

redundancy_demo = TraditionalDataParallelOverhead(model_size_gb=28, world_size=8)
redundancy = redundancy_demo.calculate_redundancy()
```

The numbers are brutal. With 8 GPUs, you're storing 7x more data than necessary. ZeRO looked at this and said: "What if we didn't?"

## ZeRO Stage 1: Optimizer State Sharding

ZeRO's first insight is simple: why store identical optimizer states on every GPU? Instead, shard them. GPU 0 gets the first chunk of Adam states, GPU 1 gets the second chunk, and so on. During the optimizer step, each GPU updates its shard, then broadcasts the updated parameters.

```python
class ZeROStage1Optimizer:
    """Simplified ZeRO Stage 1: optimizer state sharding"""
    
    def __init__(self, model: nn.Module, lr=1e-4, world_size=None, rank=None):
        self.model = model
        self.lr = lr
        self.world_size = world_size or dist.get_world_size()
        self.rank = rank or dist.get_rank()
        
        # Shard parameters across processes
        self.param_shards = self.create_parameter_shards()
        
        # Only store optimizer states for our shard
        self.momentum_buffers = {}
        self.variance_buffers = {}
        
        self.step_count = 0
        
    def create_parameter_shards(self) -> List[nn.Parameter]:
        """Divide parameters across processes"""
        
        all_params = list(self.model.parameters())
        params_per_shard = len(all_params) // self.world_size
        
        start_idx = self.rank * params_per_shard
        if self.rank == self.world_size - 1:
            # Last rank gets remaining parameters
            end_idx = len(all_params)
        else:
            end_idx = start_idx + params_per_shard
            
        my_params = all_params[start_idx:end_idx]
        
        print(f"Rank {self.rank}: managing {len(my_params)} parameters")
        return my_params
        
    def zero_grad(self):
        """Zero gradients for our parameter shard"""
        for param in self.param_shards:
            if param.grad is not None:
                param.grad.zero_()
                
    def step(self):
        """Optimizer step with parameter sharing"""
        
        self.step_count += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Update our shard of parameters
        for param in self.param_shards:
            if param.grad is None:
                continue
                
            param_id = id(param)
            
            # Initialize buffers if needed
            if param_id not in self.momentum_buffers:
                self.momentum_buffers[param_id] = torch.zeros_like(param)
                self.variance_buffers[param_id] = torch.zeros_like(param)
                
            momentum = self.momentum_buffers[param_id]
            variance = self.variance_buffers[param_id]
            
            # Adam update
            momentum.mul_(beta1).add_(param.grad, alpha=1-beta1)
            variance.mul_(beta2).addcmul_(param.grad, param.grad, value=1-beta2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** self.step_count
            bias_correction2 = 1 - beta2 ** self.step_count
            
            # Parameter update
            denom = (variance.sqrt() / np.sqrt(bias_correction2)).add_(eps)
            step_size = self.lr / bias_correction1
            param.add_(momentum / denom, alpha=-step_size)
            
        # Broadcast updated parameters to all processes
        self.broadcast_parameters()
        
    def broadcast_parameters(self):
        """Share updated parameters across all processes"""
        
        # Each process broadcasts its parameter shard to everyone
        all_params = list(self.model.parameters())
        params_per_shard = len(all_params) // self.world_size
        
        for source_rank in range(self.world_size):
            start_idx = source_rank * params_per_shard
            if source_rank == self.world_size - 1:
                end_idx = len(all_params)
            else:
                end_idx = start_idx + params_per_shard
                
            # Broadcast this rank's parameters
            for param in all_params[start_idx:end_idx]:
                dist.broadcast(param.data, src=source_rank)
                
    def get_memory_usage(self):
        """Calculate memory usage for this process"""
        
        param_memory = sum(p.numel() * 4 for p in self.param_shards)  # FP32
        momentum_memory = sum(buf.numel() * 4 for buf in self.momentum_buffers.values())
        variance_memory = sum(buf.numel() * 4 for buf in self.variance_buffers.values())
        
        total_memory = param_memory + momentum_memory + variance_memory
        
        return {
            'parameters_gb': param_memory / 1e9,
            'momentum_gb': momentum_memory / 1e9,
            'variance_gb': variance_memory / 1e9,
            'total_gb': total_memory / 1e9
        }

class ZeROMemoryComparison:
    """Compare memory usage: traditional vs ZeRO"""
    
    def __init__(self, model_params=7e9, world_size=8):
        self.model_params = model_params
        self.world_size = world_size
        
    def compare_memory_usage(self):
        """Compare traditional DP vs ZeRO Stage 1"""
        
        # Memory per parameter (FP32)
        bytes_per_param = 4
        
        # Traditional data parallel (per GPU)
        traditional_per_gpu = {
            'parameters': self.model_params * bytes_per_param,
            'gradients': self.model_params * bytes_per_param,
            'momentum': self.model_params * bytes_per_param,
            'variance': self.model_params * bytes_per_param
        }
        traditional_total = sum(traditional_per_gpu.values()) / 1e9
        
        # ZeRO Stage 1 (per GPU)
        zero_per_gpu = {
            'parameters': self.model_params * bytes_per_param,  # Still replicated
            'gradients': self.model_params * bytes_per_param,  # Still replicated
            'momentum': (self.model_params / self.world_size) * bytes_per_param,  # Sharded!
            'variance': (self.model_params / self.world_size) * bytes_per_param   # Sharded!
        }
        zero_total = sum(zero_per_gpu.values()) / 1e9
        
        memory_savings = (traditional_total - zero_total) / traditional_total
        
        print(f"Memory Comparison (7B model, {self.world_size} GPUs):")
        print(f"  Traditional DP: {traditional_total:.1f}GB per GPU")
        print(f"  ZeRO Stage 1: {zero_total:.1f}GB per GPU")
        print(f"  Memory savings: {memory_savings:.1%}")
        
        return memory_savings

comparison = ZeROMemoryComparison()
stage1_savings = comparison.compare_memory_usage()
```

ZeRO Stage 1 alone cuts optimizer memory by 50%. Not bad for a first attempt.

## ZeRO Stage 2: Gradient Sharding

Stage 1 was just the warm-up. Stage 2 realizes that gradients don't need to be replicated either. Instead of every GPU storing the full gradient tensor, each GPU accumulates only the gradients for its parameter shard. During AllReduce, we reduce-scatter instead of all-reduce.

```python
class ZeROStage2Optimizer:
    """ZeRO Stage 2: optimizer state + gradient sharding"""
    
    def __init__(self, model: nn.Module, lr=1e-4):
        self.model = model
        self.lr = lr
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Parameter and gradient sharding
        self.param_shards = self.create_parameter_shards()
        self.gradient_shards = {id(p): None for p in self.param_shards}
        
        # Optimizer states (only for our shard)
        self.momentum_buffers = {}
        self.variance_buffers = {}
        
        # Hook gradients for reduce-scatter
        self.register_gradient_hooks()
        
    def register_gradient_hooks(self):
        """Register hooks for gradient reduce-scatter"""
        
        def reduce_scatter_hook(param):
            def hook(grad):
                # Only accumulate gradients for our parameter shard
                if id(param) in self.gradient_shards:
                    if self.gradient_shards[id(param)] is None:
                        self.gradient_shards[id(param)] = grad.clone()
                    else:
                        self.gradient_shards[id(param)].add_(grad)
                        
                # Clear the full gradient to save memory
                return torch.zeros_like(grad)
            
            return hook
            
        # Only hook our parameters
        for param in self.param_shards:
            param.register_hook(reduce_scatter_hook(param))
            
    def create_parameter_shards(self):
        """Divide parameters across processes"""
        all_params = list(self.model.parameters())
        params_per_shard = len(all_params) // self.world_size
        
        start_idx = self.rank * params_per_shard
        if self.rank == self.world_size - 1:
            end_idx = len(all_params)
        else:
            end_idx = start_idx + params_per_shard
            
        return all_params[start_idx:end_idx]
        
    def reduce_scatter_gradients(self):
        """Perform reduce-scatter on gradients"""
        
        # Collect gradients from all processes for our parameters
        for param in self.param_shards:
            param_id = id(param)
            local_grad = self.gradient_shards[param_id]
            
            if local_grad is not None:
                # AllReduce this gradient across all processes
                dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)
                
                # Average by world size
                local_grad.div_(self.world_size)
                
                # Update parameter grad for optimizer
                param.grad = local_grad
                
    def step(self):
        """Optimizer step with gradient and state sharding"""
        
        # Reduce-scatter gradients first
        self.reduce_scatter_gradients()
        
        # Standard Adam update on our shard
        self.step_count += 1
        beta1, beta2 = 0.9, 0.999
        
        for param in self.param_shards:
            if param.grad is None:
                continue
                
            param_id = id(param)
            
            if param_id not in self.momentum_buffers:
                self.momentum_buffers[param_id] = torch.zeros_like(param)
                self.variance_buffers[param_id] = torch.zeros_like(param)
                
            # Adam update (same as before)
            momentum = self.momentum_buffers[param_id]
            variance = self.variance_buffers[param_id]
            
            momentum.mul_(beta1).add_(param.grad, alpha=1-beta1)
            variance.mul_(beta2).addcmul_(param.grad, param.grad, value=1-beta2)
            
            bias_correction1 = 1 - beta1 ** self.step_count
            bias_correction2 = 1 - beta2 ** self.step_count
            
            denom = (variance.sqrt() / np.sqrt(bias_correction2)).add_(1e-8)
            step_size = self.lr / bias_correction1
            param.add_(momentum / denom, alpha=-step_size)
            
        # Broadcast updated parameters
        self.broadcast_parameters()
        
        # Clear gradient shards for next iteration
        for param_id in self.gradient_shards:
            self.gradient_shards[param_id] = None
            
    def broadcast_parameters(self):
        """Broadcast updated parameters to all processes"""
        all_params = list(self.model.parameters())
        params_per_shard = len(all_params) // self.world_size
        
        for source_rank in range(self.world_size):
            start_idx = source_rank * params_per_shard
            if source_rank == self.world_size - 1:
                end_idx = len(all_params)
            else:
                end_idx = start_idx + params_per_shard
                
            for param in all_params[start_idx:end_idx]:
                dist.broadcast(param.data, src=source_rank)

# Update memory comparison for Stage 2
class ZeROStage2MemoryComparison(ZeROMemoryComparison):
    """Memory comparison including gradient sharding"""
    
    def compare_all_stages(self):
        """Compare traditional DP vs ZeRO Stage 1 vs Stage 2"""
        
        bytes_per_param = 4
        
        # Traditional DP (per GPU)
        traditional_total = (self.model_params * 4 * bytes_per_param) / 1e9  # params + grads + momentum + variance
        
        # ZeRO Stage 1 (per GPU)
        stage1_total = (self.model_params * 2 + self.model_params / self.world_size * 2) * bytes_per_param / 1e9
        
        # ZeRO Stage 2 (per GPU) 
        stage2_total = (self.model_params + self.model_params / self.world_size * 3) * bytes_per_param / 1e9
        
        print(f"Complete Memory Comparison (7B model, {self.world_size} GPUs):")
        print(f"  Traditional DP: {traditional_total:.1f}GB per GPU")
        print(f"  ZeRO Stage 1: {stage1_total:.1f}GB per GPU ({(traditional_total-stage1_total)/traditional_total:.1%} savings)")
        print(f"  ZeRO Stage 2: {stage2_total:.1f}GB per GPU ({(traditional_total-stage2_total)/traditional_total:.1%} savings)")
        
        return stage1_total, stage2_total

stage2_comparison = ZeROStage2MemoryComparison()
stage1_mem, stage2_mem = stage2_comparison.compare_all_stages()
```

Stage 2 pushes us to 75% memory savings. We're getting somewhere.

## ZeRO Stage 3: The Full Monty

Stage 3 is where ZeRO gets aggressive: even model parameters get sharded. Each GPU only stores its slice of the model weights. During forward/backward passes, parameters are gathered just-in-time, used, then discarded. It's like a library where books exist only when someone's reading them.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

class ZeROStage3Demo:
    """Demonstrate ZeRO Stage 3 with FSDP"""
    
    def __init__(self, model: nn.Module, min_num_params=1e6):
        # FSDP implements ZeRO Stage 3
        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=int(min_num_params))
        
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=None,  # Can be configured
            backward_prefetch=FSDP.BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
        )
        
    def get_memory_footprint(self):
        """Get actual memory usage with FSDP"""
        
        # FSDP handles memory management automatically
        param_memory = sum(p.numel() * 4 for p in self.model.parameters() if p.requires_grad) / 1e9
        
        # With FSDP, each process only stores its shard
        world_size = dist.get_world_size()
        sharded_memory = param_memory / world_size
        
        return {
            'total_params_gb': param_memory,
            'per_process_gb': sharded_memory,
            'memory_efficiency': world_size
        }

# Memory comparison for all stages
def final_memory_comparison():
    """Final comparison of all ZeRO stages"""
    
    model_params = 7e9
    world_size = 8
    bytes_per_param = 4
    
    traditional = model_params * 4 * bytes_per_param / 1e9  # Full replication
    stage1 = (model_params * 2 + model_params / world_size * 2) * bytes_per_param / 1e9
    stage2 = (model_params + model_params / world_size * 3) * bytes_per_param / 1e9  
    stage3 = (model_params / world_size * 4) * bytes_per_param / 1e9  # Everything sharded
    
    print(f"ZeRO Memory Efficiency (7B model, 8 GPUs):")
    print(f"  Traditional: {traditional:.1f}GB per GPU (1.0x)")
    print(f"  Stage 1:     {stage1:.1f}GB per GPU ({traditional/stage1:.1f}x better)")
    print(f"  Stage 2:     {stage2:.1f}GB per GPU ({traditional/stage2:.1f}x better)")
    print(f"  Stage 3:     {stage3:.1f}GB per GPU ({traditional/stage3:.1f}x better)")
    
    return stage3

stage3_memory = final_memory_comparison()
```

Stage 3 gives us 8x memory efficiency. That 7B model that needed 112GB per GPU? Now it needs 14GB. The math works.

## Production ZeRO: Using DeepSpeed

Real-world ZeRO is handled by DeepSpeed, which provides production implementations of all three stages:

```python
# DeepSpeed ZeRO configuration example
zero_config = {
    "zero_optimization": {
        "stage": 2,  # or 3 for maximum memory savings
        "offload_optimizer": {
            "device": "cpu",  # Offload to CPU memory
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",  # Stage 3 can offload parameters too
            "pin_memory": True
        },
        "overlap_comm": True,  # Overlap communication with computation
        "contiguous_gradients": True,  # Memory layout optimization
        "reduce_bucket_size": 5e8,  # Gradient bucketing
        "stage3_prefetch_bucket_size": 5e8,  # Stage 3 prefetching
        "stage3_param_persistence_threshold": 1e6,  # Keep small params in memory
    }
}

# Initialize with DeepSpeed
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_params=zero_config,
    optimizer=torch.optim.AdamW(model.parameters())
)

# Training loop remains largely unchanged
for batch in dataloader:
    outputs = model_engine(batch)
    loss = compute_loss(outputs)
    model_engine.backward(loss)
    model_engine.step()
```

## The ZeRO Philosophy

ZeRO represents a fundamental shift in distributed training philosophy. Instead of "every process has everything" (data parallelism) or "every process has a different part of the computation" (model parallelism), ZeRO says "every process has a different part of the state."

It's the realization that redundancy is expensive, and cleverly managed sharing can give us the benefits of both memory efficiency and computational parallelism. The tradeoff is complexity – parameters need to be gathered and scattered dynamically, communication patterns become more sophisticated, and debugging gets harder.

But the payoff is enormous: training models that simply wouldn't fit any other way.

## The Practical Reality

**ZeRO Stage 1**: Easy win, 2-4x memory savings, minimal complexity
**ZeRO Stage 2**: Better savings (4-8x), more complex communication  
**ZeRO Stage 3**: Maximum savings (8x+), but careful tuning required

The sweet spot for most applications is Stage 2 with CPU offloading. Stage 3 shines when training truly massive models where even sharded optimizer states don't fit in GPU memory.

Memory is the new compute bottleneck in deep learning. ZeRO's insight – that we can shard state without sacrificing parallelism – has become foundational to training large models efficiently. Every major framework now has some variant of ZeRO's approach.

The universal Turing machine assumed infinite memory for computation. We've discovered that carefully managed, finite memory can be more efficient than abundant, wasteful memory. Sometimes the best way to store everything is to ensure that nobody stores everything.

Next up: **Communication Backends: NCCL Setup and Optimization** – our final post in this chapter, where we dive into the networking stack that makes all this distributed coordination actually work.

ZeRO memory horror stories? Successful Stage 3 deployments? Drop them in the comments. Nothing bonds engineers like shared trauma over memory allocation strategies.
