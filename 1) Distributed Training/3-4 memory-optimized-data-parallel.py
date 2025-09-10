class MemoryOptimizedDataParallel:
    """Advanced memory optimizations for data parallel training"""
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # ZeRO-style optimizer state sharding
        self.shard_optimizer_states = True
        
        # Gradient checkpointing to trade compute for memory
        self.use_gradient_checkpointing = True
        
        # Mixed precision to halve memory usage
        self.use_mixed_precision = True
        
        self.setup_memory_optimizations(model)
        
    def setup_memory_optimizations(self, model):
        """Apply various memory optimization techniques"""
        
        # 1. Mixed precision training
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            
        # 2. Gradient checkpointing
        if self.use_gradient_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
            )
            
            # Wrap every few layers with checkpointing
            for i, layer in enumerate(model.layers):
                if i % 4 == 0:  # Checkpoint every 4th layer
                    model.layers[i] = checkpoint_wrapper(
                        layer, 
                        checkpoint_impl=CheckpointImpl.REENTRANT
                    )
        
        # 3. Parameter sharding (ZeRO-style)
        if self.shard_optimizer_states:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.model = FSDP(model)
        else:
            self.model = DDP(model, device_ids=[self.rank])
            
        self.optimizer = self.create_sharded_optimizer()
        
    def create_sharded_optimizer(self):
        """Create optimizer with sharded states"""
        
        if self.shard_optimizer_states:
            # Each GPU only stores optimizer states for its shard
            from torch.distributed.optim import ZeroRedundancyOptimizer
            
            optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=1e-4,
                # Optimizer states are sharded across GPUs
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
        return optimizer
        
    def memory_efficient_training_step(self, batch):
        """Training step with all memory optimizations enabled"""
        
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            output = self.model(batch)
            loss = compute_loss(output, targets)
            
        # Gradient scaling for mixed precision
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            
            # Clip gradients before unscaling
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        self.optimizer.zero_grad()
        
        return loss.item()
