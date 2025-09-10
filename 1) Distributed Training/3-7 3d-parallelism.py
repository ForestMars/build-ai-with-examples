class ThreeDimensionalParallelism:
    """Preview of 3D parallelism combining all strategies"""
    
    def __init__(self, 
                 data_parallel_size: int,
                 tensor_parallel_size: int, 
                 pipeline_parallel_size: int):
        
        self.dp_size = data_parallel_size
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        
        # Total world size must match
        total_gpus = data_parallel_size * tensor_parallel_size * pipeline_parallel_size
        assert total_gpus == dist.get_world_size()
        
        # Calculate this GPU's position in 3D grid
        self.rank = dist.get_rank()
        self.dp_rank = self.rank // (self.tp_size * self.pp_size)
        self.tp_rank = (self.rank // self.pp_size) % self.tp_size
        self.pp_rank = self.rank % self.pp_size
        
        print(f"Rank {self.rank}: DP={self.dp_rank}, TP={self.tp_rank}, PP={self.pp_rank}")
        
        # Create process groups for each dimension
        self.setup_process_groups()
        
    def setup_process_groups(self):
        """Create separate communication groups for each parallelism dimension"""
        
        # Data parallel groups: GPUs with same TP/PP coordinates
        self.dp_group = None
        dp_ranks = []
        for dp_rank in range(self.dp_size):
            rank = dp_rank * (self.tp_size * self.pp_size) + self.tp_rank * self.pp_size + self.pp_rank
            dp_ranks.append(rank)
            
        if dist.get_rank() in dp_ranks:
            self.dp_group = dist.new_group(dp_ranks)
            
        # Similar setup for TP and PP groups...
        # (This is getting complex - that's the point!)
