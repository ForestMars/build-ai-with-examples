import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class DataParallelTrainingStep:
    """The canonical data parallel training pattern"""
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # Each GPU gets its own copy of the model
        self.model = model.cuda(rank)
        
        # Wrap with DDP for automatic gradient synchronization
        self.ddp_model = DDP(self.model, device_ids=[rank])
        
        self.optimizer = torch.optim.Adam(self.ddp_model.parameters())
        
    def training_step(self, batch):
        """Single training step with gradient synchronization"""
        
        # 1. SCATTER: Each GPU gets its slice of the batch
        local_batch = batch[self.rank::self.world_size]  # Simple striping
        
        # 2. COMPUTE: Forward and backward pass (completely independent)
        self.optimizer.zero_grad()
        output = self.ddp_model(local_batch)
        loss = compute_loss(output, targets)
        loss.backward()  # This triggers gradient synchronization!
        
        # 3. GATHER: AllReduce happens automatically inside DDP
        # Gradients are averaged across all GPUs
        
        # 4. UPDATE: Everyone applies the same averaged gradients
        self.optimizer.step()
        
        return loss.item()
