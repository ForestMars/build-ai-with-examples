from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, CheckpointWrapper
)

class ProductionCheckpointer:
    """Production-grade gradient checkpointing"""
    
    def __init__(self, checkpoint_ratio=0.25, mixed_precision=True):
        self.checkpoint_ratio = checkpoint_ratio
        self.mixed_precision = mixed_precision
        
    def apply_checkpointing(self, model: nn.Module, layer_pattern='layers'):
        """Apply checkpointing to model layers"""
        
        # Find layers to checkpoint
        layers = getattr(model, layer_pattern)
        num_layers = len(layers)
        checkpoint_every = max(1, int(1 / self.checkpoint_ratio))
        
        print(f"Applying checkpointing every {checkpoint_every} layers ({num_layers} total)")
        
        # Wrap selected layers
        for i in range(0, num_layers, checkpoint_every):
            if i < num_layers:
                layers[i] = checkpoint_wrapper(
                    layers[i],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
                
        return model
        
    def memory_efficient_training_step(self, model, batch, targets, optimizer, scaler=None):
        """Training step optimized for checkpointing"""
        
        optimizer.zero_grad()
        
        if self.mixed_precision and scaler:
            from torch.cuda.amp import autocast
            
            with autocast():
                outputs = model(batch)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            
        return loss.item()

# Quick demo
model = SimpleTransformer(num_layers=16, d_model=1024).cuda()
checkpointer = ProductionCheckpointer(checkpoint_ratio=0.25)
checkpointed_model = checkpointer.apply_checkpointing(model)

print("Gradient checkpointing applied successfully!")



