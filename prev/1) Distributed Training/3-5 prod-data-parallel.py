import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import time
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DataParallelConfig:
    # Memory optimizations
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_parameter_sharding: bool = True  # FSDP vs DDP
    
    # Communication optimizations  
    gradient_accumulation_steps: int = 4
    use_hierarchical_allreduce: bool = True
    overlap_communication: bool = True
    
    # Fault tolerance
    checkpoint_every_n_steps: int = 1000
    max_failures: int = 3
    
    # Performance tuning
    bucket_cap_mb: int = 25  # DDP gradient bucketing
    find_unused_parameters: bool = False

class ProductionDataParallel:
    """Production-ready data parallel training with all optimizations"""
    
    def __init__(self, 
                 model: nn.Module, 
                 rank: int, 
                 world_size: int,
                 config: DataParallelConfig):
        
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        # Setup model with memory optimizations
        self.setup_model(model)
        
        # Setup optimizer with sharding
        self.setup_optimizer()
        
        # Setup mixed precision
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            
        # Performance tracking
        self.step_times = []
        self.communication_times = []
        self.compute_times = []
        
        # Fault tolerance
        self.failure_count = 0
        self.last_checkpoint_step = 0
        
    def setup_model(self, model: nn.Module):
        """Setup model with appropriate parallelism strategy"""
        
        # Move to GPU
        model = model.cuda(self.rank)
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing(model)
            
        # Choose parallelism strategy
        if self.config.use_parameter_sharding:
            # FSDP for memory efficiency with large models
            self.model = FSDP(
                model,
                auto_wrap_policy=self.get_fsdp_wrap_policy(),
                mixed_precision=self.get_mixed_precision_policy(),
                backward_prefetch=FSDP.BackwardPrefetch.BACKWARD_PRE,
                forward_prefetch=True,
                limit_all_gathers=True,
            )
        else:
            # DDP for maximum performance with smaller models
            self.model = DDP(
                model, 
                device_ids=[self.rank],
                bucket_cap_mb=self.config.bucket_cap_mb,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=False,  # Save communication
            )
            
    def setup_optimizer(self):
        """Setup optimizer with optional sharding"""
        
        if self.config.use_parameter_sharding:
            # FSDP handles optimizer sharding automatically
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            # Optional ZeRO-style optimizer sharding with DDP
            from torch.distributed.optim import ZeroRedundancyOptimizer
            
            self.optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=1e-4,
                weight_decay=0.01
            )
            
    def training_step(self, batch, step: int) -> float:
        """Optimized training step with all features"""
        
        step_start = time.perf_counter()
        
        try:
            # Gradient accumulation loop
            total_loss = 0.0
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                # Get micro-batch
                micro_batch = self.get_micro_batch(batch, micro_step)
                
                # Forward pass with mixed precision
                compute_start = time.perf_counter()
                
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(micro_batch)
                    loss = compute_loss(output) / self.config.gradient_accumulation_steps
                    
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                total_loss += loss.item()
                
                compute_time = time.perf_counter() - compute_start
                self.compute_times.append(compute_time)
                
            # Communication phase
            comm_start = time.perf_counter()
            
            # Gradient clipping before optimizer step
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step (triggers AllReduce)
            if self.config.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            comm_time = time.perf_counter() - comm_start
            self.communication_times.append(comm_time)
            
            # Checkpointing
            if step % self.config.checkpoint_every_n_steps == 0:
                self.save_checkpoint(step, total_loss)
                
            step_time = time.perf_counter() - step_start
            self.step_times.append(step_time)
            
            return total_loss
            
        except Exception as e:
            self.handle_failure(e, step)
            raise
            
    def get_micro_batch(self, batch, micro_step: int):
        """Split batch into micro-batches for gradient accumulation"""
        
        batch_size = batch.shape[0]
        micro_batch_size = batch_size // self.config.gradient_accumulation_steps
        
        start_idx = micro_step * micro_batch_size
        end_idx = start_idx + micro_batch_size
        
        return batch[start_idx:end_idx]
        
    def print_performance_stats(self):
        """Print detailed performance statistics"""
        
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            avg_compute_time = sum(self.compute_times) / len(self.compute_times)
            avg_comm_time = sum(self.communication_times) / len(self.communication_times)
            
            # Calculate efficiency metrics
            compute_ratio = avg_compute_time / avg_step_time
            comm_ratio = avg_comm_time / avg_step_time
            
            print(f"\nRank {self.rank} Performance Stats:")
            print(f"  Average step time:    {avg_step_time*1000:.1f}ms")
            print(f"  Average compute time: {avg_compute_time*1000:.1f}ms ({compute_ratio:.1%})")
            print(f"  Average comm time:    {avg_comm_time*1000:.1f}ms ({comm_ratio:.1%})")
            print(f"  Compute efficiency:   {compute_ratio:.1%}")
            print(f"  Communication overhead: {comm_ratio:.1%}")
            
            # Model throughput
            if hasattr(self, 'tokens_per_step'):
                throughput = self.tokens_per_step / avg_step_time
                print(f"  Throughput: {throughput:.0f} tokens/second")

def main_training_function(rank: int, world_size: int):
    """Main training function for data parallel setup"""
    
    # Initialize distributed training
    setup_distributed(rank, world_size)
    
    # Create model and training setup
    model = YourTransformerModel()
    config = DataParallelConfig()
    
    trainer = ProductionDataParallel(model, rank, world_size, config)
    
    # Create data loader with proper sharding
    dataset = YourDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Training loop
    trainer.model.train()
    global_step = 0
    
    try:
        # Try to load existing checkpoint
        if os.path.exists("latest_checkpoint.pt"):
            global_step = trainer.load_checkpoint("latest_checkpoint.pt")
            print(f"Rank {rank}: Resumed from step {global_step}")
    except Exception as e:
        print(f"Rank {rank}: Could not load checkpoint: {e}")
        
    for epoch in range(100):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to GPU
            batch = batch.cuda(rank, non_blocking=True)
            
            # Training step
            loss = trainer.training_step(batch, global_step)
            
            if rank == 0 and global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss:.4f}")
                
            global_step += 1
            
            # Break early for demonstration
            if global_step >= 10000:
                break
                
    # Print final performance statistics
    trainer.print_performance_stats()
    
    # Cleanup
    cleanup_distributed()

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

# Launch script
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Launching training on {world_size} GPUs")
    
    mp.spawn(main_training_function, args=(world_size,), nprocs=world_size, join=True)
