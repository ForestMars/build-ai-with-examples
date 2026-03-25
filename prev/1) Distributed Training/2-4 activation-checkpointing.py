class ActivationCheckpointing:
    """Manage activation storage with checkpointing to save memory"""
    
    def __init__(self, checkpoint_every_n_layers: int = 4):
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.activation_cache = {}
        self.memory_pool = []
        
    def should_checkpoint(self, layer_id: int) -> bool:
        """Decide whether to checkpoint this layer's activations"""
        return layer_id % self.checkpoint_every_n_layers == 0
        
    def store_activation(self, layer_id: int, micro_batch_id: int, activation: torch.Tensor):
        """Store activation with optional checkpointing"""
        key = (layer_id, micro_batch_id)
        
        if self.should_checkpoint(layer_id):
            # Store on CPU to save GPU memory
            self.activation_cache[key] = activation.cpu()
        else:
            # Keep on GPU for faster access
            self.activation_cache[key] = activation
            
    def retrieve_activation(self, layer_id: int, micro_batch_id: int) -> torch.Tensor:
        """Retrieve activation, recomputing if necessary"""
        key = (layer_id, micro_batch_id)
        
        if key in self.activation_cache:
            activation = self.activation_cache[key]
            if activation.device.type == 'cpu':
                activation = activation.cuda()
            del self.activation_cache[key]  # Free memory immediately
            return activation
        else:
            # Recompute from last checkpoint
            return self.recompute_activation(layer_id, micro_batch_id)
            
    def recompute_activation(self, layer_id: int, micro_batch_id: int) -> torch.Tensor:
        """Recompute activation from the last checkpoint"""
        # Find the last checkpointed layer
        checkpoint_layer = (layer_id // self.checkpoint_every_n_layers) * self.checkpoint_every_n_layers
        
        # Retrieve checkpoint and recompute forward
        checkpoint = self.retrieve_activation(checkpoint_layer, micro_batch_id)
        
        # Forward through layers until we reach target layer
        x = checkpoint
        for i in range(checkpoint_layer + 1, layer_id + 1):
            x = self.model.layers[i](x)
            
        return x
