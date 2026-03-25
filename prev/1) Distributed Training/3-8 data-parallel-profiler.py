class DataParallelProfiler:
    """Comprehensive profiling for data parallel training"""
    
    def __init__(self):
        self.communication_log = []
        self.compute_log = []
        self.memory_log = []
        
    def profile_allreduce_performance(self, model: DDP):
        """Profile AllReduce communication patterns"""
        
        # Hook into DDP's communication
        def allreduce_hook(bucket):
            start_time = time.perf_counter()
            
            # Let DDP do its thing
            result = bucket.buffer()
            
            end_time = time.perf_counter()
            comm_time = end_time - start_time
            
            self.communication_log.append({
                'bucket_size': bucket.buffer().numel() * 4,  # bytes
                'communication_time': comm_time,
                'bandwidth': bucket.buffer().numel() * 4 / comm_time / 1e9,  # GB/s
            })
            
            return result
            
        # Register hooks
        model.register_comm_hook(state=None, hook=allreduce_hook)
        
    def print_communication_analysis(self):
        """Analyze communication patterns and bottlenecks"""
        
        if not self.communication_log:
            print("No communication data collected")

            return
            
        total_bytes = sum(log['bucket_size'] for log in self.communication_log)
        total_time = sum(log['communication_time'] for log in self.communication_log)
        avg_bandwidth = sum(log['bandwidth'] for log in self.communication_log) / len(self.communication_log)
        
        print("AllReduce Performance Analysis:")
        print(f"  Total communication: {total_bytes / 1e9:.2f} GB")
        print(f"  Total comm time: {total_time * 1000:.1f} ms")
        print(f"  Average bandwidth: {avg_bandwidth:.1f} GB/s")
        
        # Identify bottlenecks
        slow_buckets = [log for log in self.communication_log if log['bandwidth'] < avg_bandwidth * 0.5]
        if slow_buckets:
            print(f"  WARNING: {len(slow_buckets)} slow communication buckets detected")
            print(f"    Slowest bandwidth: {min(log['bandwidth'] for log in slow_buckets):.1f} GB/s")
