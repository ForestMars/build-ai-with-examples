class GradientSynchronizationStrategies:
    """Different approaches to gradient synchronization timing"""
    
    def __init__(self, model: nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
        
        # Track gradient accumulation
        self.gradient_accumulation_steps = 4
        self.accumulated_steps = 0
        
    def synchronous_updates(self, loss):
        """Traditional approach: sync after every backward pass"""
        loss.backward()  # AllReduce happens here
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Pros: Simple, deterministic
        # Cons: Communication overhead on every step
        
    def gradient_accumulation_sync(self, loss):
        """Sync less frequently by accumulating gradients"""
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_steps += 1
        
        if self.accumulated_steps == self.gradient_accumulation_steps:
            # Only sync when we're ready to update
            self.sync_gradients()  # Manual AllReduce
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_steps = 0
            
        # Pros: Fewer AllReduce calls, larger effective batch size
        # Cons: Delayed updates, more memory usage
        
    def async_gradient_updates(self, loss):
        """Overlap computation with communication"""
        
        # Compute gradients locally
        loss.backward()  # No automatic sync
        
        # Start async AllReduce
        handles = []
        for param in self.model.parameters():
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, async_op=True)
                handles.append(handle)
                
        # Do other work while communication happens
        self.compute_other_metrics()
        self.log_training_stats()
        
        # Wait for communication to complete
        for handle in handles:
            handle.wait()
            
        # Apply averaged gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= self.world_size
                
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Pros: Better overlap of computation and communication
        # Cons: More complex, requires careful orchestration
