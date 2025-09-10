class FaultTolerantDataParallel:
    """Data parallel training with fault tolerance and checkpointing"""
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model = DDP(model, device_ids=[rank])
        
        # Fault tolerance configuration
        self.checkpoint_every_n_steps = 1000
        self.max_failures = 3
        self.failure_count = 0
        
        # Elastic training support
        self.min_world_size = world_size // 2
        self.max_world_size = world_size * 2
        
    def robust_training_step(self, batch, step: int):
        """Training step with failure recovery"""
        
        try:
            # Normal training step
            loss = self.training_step(batch)
            
            # Periodic checkpointing
            if step % self.checkpoint_every_n_steps == 0:
                self.save_checkpoint(step)
                
            # Reset failure counter on success
            self.failure_count = 0
            
            return loss
            
        except (RuntimeError, dist.DistBackendError) as e:
            # Handle distributed training failures
            self.handle_training_failure(e, step)
            
    def handle_training_failure(self, error: Exception, step: int):
        """Recover from training failures"""
        
        self.failure_count += 1
        
        if self.failure_count > self.max_failures:
            print(f"Rank {self.rank}: Too many failures, giving up")
            raise error
            
        print(f"Rank {self.rank}: Training failure at step {step}: {error}")
        
        # Try to recover
        if "NCCL" in str(error):
            self.recover_nccl_failure()
        elif "CUDA" in str(error):
            self.recover_cuda_failure()
        else:
            self.recover_generic_failure()
            
    def recover_nccl_failure(self):
        """Recover from NCCL communication failures"""
        
        try:
            # Destroy current process group
            if dist.is_initialized():
                dist.destroy_process_group()
                
            # Wait for other processes to clean up
            time.sleep(5)
            
            # Reinitialize with same configuration
            dist.init_process_group(
                backend='nccl',
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Recreate DDP wrapper
            self.model = DDP(self.model.module, device_ids=[self.rank])
            
            print(f"Rank {self.rank}: NCCL recovery successful")
            
        except Exception as e:
            print(f"Rank {self.rank}: NCCL recovery failed: {e}")
            raise
            
    def save_checkpoint(self, step: int):
        """Save training checkpoint for recovery"""
        
        if self.rank == 0:  # Only master saves
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'world_size': self.world_size,
                'failure_count': self.failure_count,
            }
            
            checkpoint_path = f"checkpoint_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Also save latest checkpoint
            torch.save(checkpoint, "latest_checkpoint.pt")
            
        # Synchronize all ranks
        dist.barrier()
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.rank}')
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.failure_count = checkpoint.get('failure_count', 0)
        
        return checkpoint['step']
