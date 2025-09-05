from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    """Transformer with gradient checkpointing"""
    
    def __init__(self, num_layers=20, d_model=4096, checkpoint_every=4):
        super().__init__()
        self.d_model = d_model
        self.checkpoint_every = checkpoint_every
        
        # Group layers for checkpointing
        self.layer_groups = nn.ModuleList([
            nn.ModuleList([
                TransformerLayer(d_model) 
                for _ in range(min(checkpoint_every, num_layers - i))
            ])
            for i in range(0, num_layers, checkpoint_every)
        ])
        
    def forward(self, x):
        for group in self.layer_groups:
            # Checkpoint this group of layers
            x = checkpoint(self._forward_group, x, group, use_reentrant=False)
        return x
    
    def _forward_group(self, x, layer_group):
        """Forward pass through a group of layers"""
        for layer in layer_group:
            x = layer(x)

        return x

class CheckpointingBenchmark:
    """Compare memory usage with and without checkpointing"""
    
    def __init__(self):
        pass
        
    def benchmark_memory_usage(self, num_layers=40):
        """Compare checkpointed vs normal forward pass memory"""
        
        batch_size, seq_len, d_model = 8, 1024, 2048
        
        # Normal model
        normal_model = SimpleTransformer(num_layers, d_model).cuda()
        
        # Checkpointed model  
        checkpointed_model = CheckpointedTransformer(num_layers, d_model, checkpoint_every=4).cuda()
        
        x = torch.randn(batch_size, seq_len, d_model).cuda()
        
        # Benchmark normal model
        torch.cuda.reset_peak_memory_stats()
        _ = normal_model(x)
        normal_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Benchmark checkpointed model
        torch.cuda.reset_peak_memory_stats()
        _ = checkpointed_model(x)
        checkpointed_memory = torch.cuda.max_memory_allocated() / 1e9
        
        savings = (normal_memory - checkpointed_memory) / normal_memory
        
        print(f"Memory Usage Comparison ({num_layers} layers):")
        print(f"  Normal: {normal_memory:.2f}GB")
        print(f"  Checkpointed: {checkpointed_memory:.2f}GB")  
        print(f"  Savings: {savings:.1%}")
        
        return savings
        
    def benchmark_training_speed(self, num_layers=20):
        """Compare training speed with and without checkpointing"""
        
        batch_size, seq_len, d_model = 4, 512, 1024
        
        models = {
            'normal': SimpleTransformer(num_layers, d_model).cuda(),
            'checkpointed': CheckpointedTransformer(num_layers, d_model, checkpoint_every=2).cuda()
        }
        
        x = torch.randn(batch_size, seq_len, d_model).cuda()
        target = torch.randn(batch_size, seq_len, d_model).cuda()
        
        results = {}
        
        for name, model in models.items():
            optimizer = torch.optim.Adam(model.parameters())
            
            # Warmup
            for _ in range(5):
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(20):
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
                
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            
            results[name] = total_time / 20  # Per step
            
        slowdown = results['checkpointed'] / results['normal']
        
        print(f"\nTraining Speed Comparison:")
        print(f"  Normal: {results['normal']*1000:.1f}ms/step")
        print(f"  Checkpointed: {results['checkpointed']*1000:.1f}ms/step")
        print(f"  Slowdown: {slowdown:.2f}x")
        
        return slowdown

benchmark = CheckpointingBenchmark()
memory_savings = benchmark.benchmark_memory_usage(num_layers=40)
speed_cost = benchmark.benchmark_training_speed(num_layers=20)

