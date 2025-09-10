class TopologyAwareAllReduce:
    """AllReduce implementation that considers hardware topology"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.topology = self.detect_hardware_topology()
        
    def detect_hardware_topology(self) -> dict:
        """Detect GPU interconnect topology"""
        
        topology = {
            'nodes': self.get_node_count(),
            'gpus_per_node': self.get_gpus_per_node(),
            'interconnect': self.get_interconnect_type(),
            'numa_domains': self.get_numa_topology(),
        }
        
        return topology
        
    def hierarchical_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Two-stage AllReduce: local then global"""
        
        # Stage 1: AllReduce within each node (fast NVLink)
        local_rank = self.rank % self.topology['gpus_per_node']
        node_id = self.rank // self.topology['gpus_per_node']
        
        # Create local process group for this node
        local_group = self.get_local_process_group(node_id)
        
        # Local AllReduce using high-bandwidth NVLink
        dist.all_reduce(tensor, group=local_group)
        
        # Stage 2: AllReduce between nodes (slower InfiniBand)
        if local_rank == 0:  # Only one GPU per node participates
            cross_node_group = self.get_cross_node_process_group()
            dist.all_reduce(tensor, group=cross_node_group)
            
        # Stage 3: Broadcast within node
        dist.broadcast(tensor, src=node_id * self.topology['gpus_per_node'], 
                      group=local_group)
        
        return tensor
        
    def bandwidth_aware_chunk_sizing(self, tensor: torch.Tensor) -> list:
        """Adapt chunk sizes based on interconnect bandwidth"""
        
        if self.topology['interconnect'] == 'nvlink':
            # High bandwidth: larger chunks reduce latency overhead
            chunk_size = min(tensor.numel() // 4, 1024 * 1024)  # 1M elements max
        elif self.topology['interconnect'] == 'infiniband':
            # Lower bandwidth: smaller chunks for better pipelining
            chunk_size = min(tensor.numel() // 8, 256 * 1024)   # 256K elements max
        else:
            # Ethernet: very small chunks
            chunk_size = min(tensor.numel() // 16, 64 * 1024)   # 64K elements max
            
        return tensor.split(chunk_size)
        
    def get_interconnect_type(self) -> str:
        """Detect the type of GPU interconnect"""
        try:
            # Check for NVLink
            if torch.cuda.device_count() > 1:
                # Simple heuristic: measure bandwidth between GPUs
                bandwidth = self.measure_p2p_bandwidth()
                if bandwidth > 200:  # GB/s, typical NVLink speeds
                    return 'nvlink'
                elif bandwidth > 50:  # GB/s, typical InfiniBand
                    return 'infiniband'
                else:
                    return 'ethernet'
        except:
            return 'unknown'
            
    def measure_p2p_bandwidth(self) -> float:
        """Measure peer-to-peer bandwidth between GPUs"""
        if self.rank == 0 and torch.cuda.device_count() > 1:
            # Create test tensors
            size = 100 * 1024 * 1024  # 100MB
            src_tensor = torch.randn(size, device='cuda:0')
            dst_tensor = torch.empty(size, device='cuda:1')
            
            # Warmup
            for _ in range(10):
                dst_tensor.copy_(src_tensor, non_blocking=True)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(100):
                dst_tensor.copy_(src_tensor, non_blocking=True)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate bandwidth in GB/s
            bytes_transferred = size * 4 * 100  # float32 = 4 bytes
            time_taken = end_time - start_time
            bandwidth = bytes_transferred / time_taken / (1024**3)
            
            return bandwidth
        return 0.0
