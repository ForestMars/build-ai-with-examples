import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, List
import asyncio
import logging

@dataclass
class PipelineConfig:
    num_stages: int = 4
    num_micro_batches: int = 8
    checkpoint_every_n_layers: int = 4
    use_async_communication: bool = True
    profile_layers: bool = True

class ProductionPipelineStage:
    """Production-ready pipeline stage with all optimizations"""
    
    def __init__(self, 
                 layers: List[nn.Module], 
                 stage_id: int, 
                 config: PipelineConfig):
        self.layers = nn.Sequential(*layers).cuda()
        self.stage_id = stage_id
        self.config = config
        self.rank = dist.get_rank()
        
        # Communication optimizations
        self.comm = OptimizedP2PCommunication(self.rank, dist.get_world_size())

        # TODO: Define expected tensor shapes for communication buffers
        #       (From model config or by tracing layer outputs) 
        self.expected_input_shape = self._get_input_shape()
        self.expected_output_shape = self._get_output_shape()
        
        # Memory optimizations
        self.activation_checkpointer = ActivationCheckpointing(
            config.checkpoint_every_n_layers
        )
        
        # Profiling
        self.forward_times = []
        self.backward_times = []
        self.communication_times = []
        
    def forward_micro_batch(self, micro_batch_id: int, input_tensor: Optional[torch.Tensor] = None):
        """Process one micro-batch through this stage"""
        
        start_time = time.perf_counter()
        
        # Receive input from previous stage
        if self.stage_id > 0 and input_tensor is None:
            comm_start = time.perf_counter()
            req, recv_buffer = self.comm.async_recv(
                shape=self.expected_input_shape,
                dtype=torch.float16,
                src=self.rank - 1,
                tag=micro_batch_id
            )
            self.comm.wait_recv(micro_batch_id)
            #  This is a common pattern but can be inefficient as clone creates an extra mem alloc 
            #  Better would be to use recv_buffer directly since it's already on local gpu
            input_tensor = recv_buffer.clone()
            self.communication_times.append(time.perf_counter() - comm_start)
            
        # Forward pass
        output = self.layers(input_tensor)
        
        # Send to next stage
        if self.stage_id < self.config.num_stages - 1:
            comm_start = time.perf_counter()
            self.comm.async_send(output, dst=self.rank + 1, tag=micro_batch_id)
            self.communication_times.append(time.perf_counter() - comm_start)
            
        # Store for backward pass
        self.activation_checkpointer.store_activation(
            self.stage_id, micro_batch_id, input_tensor
        )
        
        forward_time = time.perf_counter() - start_time
        self.forward_times.append(forward_time)
        
        return output
    
    def backward_micro_batch(self, micro_batch_id: int, grad_output: Optional[torch.Tensor] = None):
        """Backward pass for one micro-batch"""
        
        start_time = time.perf_counter()
        
        # Receive gradient from next stage
        if self.stage_id < self.config.num_stages - 1 and grad_output is None:
            req, recv_buffer = self.comm.async_recv(
                shape=self.expected_output_shape,
                dtype=torch.float16,
                src=self.rank + 1,
                tag=micro_batch_id + 10000  # Offset to avoid conflicts
            )
            self.comm.wait_recv(micro_batch_id + 10000)
            grad_output = recv_buffer.clone()
            
        # Retrieve stored activation
        input_activation = self.activation_checkpointer.retrieve_activation(
            self.stage_id, micro_batch_id
        )
        
        # Backward pass
        input_activation.requires_grad_(True)
        output = self.layers(input_activation)
        output.backward(grad_output)
        
        # Send gradient to previous stage
        if self.stage_id > 0:
            grad_input = input_activation.grad
            self.comm.async_send(grad_input, dst=self.rank - 1, tag=micro_batch_id + 10000)
            
        backward_time = time.perf_counter() - start_time
        self.backward_times.append(backward_time)
        
    def run_pipeline_schedule(self, schedule: List[tuple]):
        """Execute the pipeline schedule"""
        
        for operation, stage_id, micro_batch_id in schedule:
            if stage_id == self.stage_id:
                if operation == 'forward':
                    self.forward_micro_batch(micro_batch_id)
                elif operation == 'backward':
                    self.backward_micro_batch(micro_batch_id)
                    
        # Wait for all communication to complete
        for tag in list(self.comm.pending_sends.keys()):
            self.comm.wait_send(tag)
            
    def print_profiling_stats(self):
        """Print performance statistics"""
        if self.forward_times:
            avg_forward = sum(self.forward_times) / len(self.forward_times)
            avg_backward = sum(self.backward_times) / len(self.backward_times)
            avg_comm = sum(self.communication_times) / len(self.communication_times)
            
            print(f"Stage {self.stage_id} Performance:")
            print(f"  Average Forward:  {avg_forward*1000:.2f}ms")
            print(f"  Average Backward: {avg_backward*1000:.2f}ms")
            print(f"  Average Comm:     {avg_comm*1000:.2f}ms")
            print(f"  Bubble Time:      {self.estimate_bubble_time():.1f}%")
            
    def estimate_bubble_time(self) -> float:
        """Estimate percentage of time spent in pipeline bubbles"""
        if not self.forward_times:

            return 0.0
            
        total_computation = sum(self.forward_times) + sum(self.backward_times)
        total_communication = sum(self.communication_times)
        
        # Rough estimate: bubbles = idle time / total time
        bubble_ratio = total_communication / (total_computation + total_communication)

        return bubble_ratio * 100

# Usage example
def train_with_pipeline_parallelism(rank, world_size):
    """Main training function with pipeline parallelism"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create pipeline configuration
    config = PipelineConfig(
        num_stages=world_size,
        num_micro_batches=16,
        checkpoint_every_n_layers=2
    )
    
    # Create transformer layers (simplified)
    total_layers = 32
    layers_per_stage = total_layers // world_size
    
    my_layers = []
    for i in range(layers_per_stage):
        layer_id = rank * layers_per_stage + i
        my_layers.append(TransformerLayer(d_model=512, layer_id=layer_id))
    
    # Create pipeline stage
    stage = ProductionPipelineStage(my_layers, rank, config)
    
    # Create schedule
    scheduler = PipelineScheduler(world_size, config.num_micro_batches)
    schedule = scheduler.create_schedule()
    
    print(f"Rank {rank}: Running {len(schedule)} pipeline operations")
    
    # Training loop
    for epoch in range(10):
        stage.run_pipeline_schedule(schedule)
        
        if rank == 0:
            print(f"Completed epoch {epoch}")
            
    # Print performance stats
    stage.print_profiling_stats()
    
    cleanup_distributed()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train_with_pipeline_parallelism, args=(world_size,), nprocs=world_size)