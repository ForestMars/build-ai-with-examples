class OptimizedP2PCommunication:
    """Optimized point-to-point communication for pipeline parallelism"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # Pre-allocate communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        
        # Track in-flight operations
        self.pending_sends = {}
        self.pending_recvs = {}
        
    def async_send(self, tensor: torch.Tensor, dst: int, tag: int):
        """Non-blocking send with buffer management"""
        # Copy to pinned memory for faster transfer
        if tag not in self.send_buffers:
            self.send_buffers[tag] = torch.empty_like(tensor, pin_memory=True)
            
        send_buffer = self.send_buffers[tag]
        send_buffer.copy_(tensor, non_blocking=True)
        
        # Start async send
        req = dist.isend(send_buffer, dst=dst, tag=tag)
        self.pending_sends[tag] = req
        
        return req
    
    def async_recv(self, shape: tuple, dtype: torch.dtype, src: int, tag: int):
        """Non-blocking receive with buffer management"""
        # Use pre-allocated pinned buffer
        if tag not in self.recv_buffers:
            self.recv_buffers[tag] = torch.empty(
                shape, dtype=dtype, pin_memory=True, device='cuda'
            )
            
        recv_buffer = self.recv_buffers[tag]
        
        # Start async receive
        req = dist.irecv(recv_buffer, src=src, tag=tag)
        self.pending_recvs[tag] = req
        
        return req, recv_buffer
    
    def wait_send(self, tag: int):
        """Wait for send to complete"""
        if tag in self.pending_sends:
            self.pending_sends[tag].wait()
            del self.pending_sends[tag]
            
    def wait_recv(self, tag: int):
        """Wait for receive to complete"""
        if tag in self.pending_recvs:
            self.pending_recvs[tag].wait()
            del self.pending_recvs[tag]
            return self.recv_buffers[tag]
