class MicroBatch:
    """A single micro-batch flowing through the pipeline"""
    
    def __init__(self, data: torch.Tensor, micro_batch_id: int):
        self.data = data
        self.micro_batch_id = micro_batch_id
        self.stage_outputs = {}  # Cache outputs at each stage
        
    def __repr__(self):
        return f"MicroBatch(id={self.micro_batch_id}, shape={self.data.shape})"

class PipelineScheduler:
    """Manages the flow of micro-batches through pipeline stages"""
    
    def __init__(self, num_stages: int, num_micro_batches: int):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        
        # Each stage maintains a queue of micro-batches
        self.stage_queues = [[] for _ in range(num_stages)]
        self.completed_forward = []
        self.completed_backward = []
        
    def create_schedule(self):
        """Generate the optimal schedule for forward/backward passes"""
        schedule = []
        
        # Forward passes: stagger micro-batches across stages
        for step in range(self.num_stages + self.num_micro_batches - 1):
            for stage_id in range(self.num_stages):
                micro_batch_id = step - stage_id
                if 0 <= micro_batch_id < self.num_micro_batches:
                    schedule.append(('forward', stage_id, micro_batch_id))
        
        # Backward passes: reverse order
        for step in range(self.num_stages + self.num_micro_batches - 1):
            for stage_id in range(self.num_stages - 1, -1, -1):
                micro_batch_id = step - (self.num_stages - 1 - stage_id)
                if 0 <= micro_batch_id < self.num_micro_batches:
                    schedule.append(('backward', stage_id, micro_batch_id))
                    
        return schedule