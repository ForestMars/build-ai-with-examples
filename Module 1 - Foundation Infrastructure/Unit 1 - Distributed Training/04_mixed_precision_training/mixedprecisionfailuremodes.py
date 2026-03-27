class MixedPrecisionFailureModes:
    """Demonstrate when mixed precision can fail"""
    
    def __init__(self):
        self.failure_examples = []
        
    def demonstrate_pathological_case(self):
        """Show a case where mixed precision struggles"""
        
        # Create a model with extreme dynamic range in parameters
        class PathologicalModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Layer with tiny weights
                self.tiny_layer = nn.Linear(100, 100)
                with torch.no_grad():
                    self.tiny_layer.weight.fill_(1e-6)
                    
                # Layer with huge weights  
                self.huge_layer = nn.Linear(100, 100)
                with torch.no_grad():
                    self.huge_layer.weight.fill_(1e3)
                    
                # Normal layer
                self.normal_layer = nn.Linear(100, 10)
                
            def forward(self, x):
                x = torch.tanh(self.tiny_layer(x))  # Gradients will be tiny
                x = torch.tanh(self.huge_layer(x))  # Activations will be saturated
                x = self.normal_layer(x)
                return x
                
        model = PathologicalModel().cuda()
        
        # Try to train with mixed precision
        scaler = GradScaler(init_scale=65536.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        overflow_count = 0
        underflow_count = 0
        
        for step in range(100):
            batch = torch.randn(32, 100).cuda()
            targets = torch.randint(0, 10, (32,)).cuda()
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(batch)
                loss = nn.CrossEntropyLoss()(output, targets)
                
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # Check for problems
            has_inf = False
            has_tiny_grads = False
            
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isinf(param.grad).any():
                        has_inf = True
                    if (param.grad.abs() < 1e-7).all():
                        has_tiny_grads = True
                        
            if has_inf:
                overflow_count += 1
            if has_tiny_grads:
                underflow_count += 1
                
            # Try to step
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
        print(f"Pathological Model Results:")
        print(f"  Overflows: {overflow_count}/100")
        print(f"  Underflows: {underflow_count}/100")
        print(f"  Final scale: {scaler.get_scale():.0f}")
        
        return overflow_count > 10 or underflow_count > 10
        
    def benchmark_small_model_overhead(self):
        """Show that mixed precision can hurt performance on small models"""
        
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(), 
                    nn.Linear(32, 10)
                )
                
            def forward(self, x):
                return self.layers(x)
                
        model = TinyModel().cuda()
        batch = torch.randn(16, 32).cuda()
        targets = torch.randint(0, 10, (16,)).cuda()
        
        # Benchmark FP32
        optimizer_fp32 = torch.optim.Adam(model.parameters())
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(1000):
            optimizer_fp32.zero_grad()
            output = model(batch)
            loss = nn.CrossEntropyLoss()(output, targets)
            loss.backward()
            optimizer_fp32.step()
            
        torch.cuda.synchronize()
        fp32_time = time.perf_counter() - start_time
        
        # Benchmark Mixed Precision
        optimizer_mp = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(1000):
            optimizer_mp.zero_grad()
            
            with autocast():
                output = model(batch)
                loss = nn.CrossEntropyLoss()(output, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_mp)
            scaler.update()
            
        torch.cuda.synchronize()
        mp_time = time.perf_counter() - start_time
        
        overhead = mp_time / fp32_time - 1.0
        
        print(f"Small Model Overhead Analysis:")
        print(f"  FP32 time: {fp32_time:.3f}s")
        print(f"  Mixed precision time: {mp_time:.3f}s")
        print(f"  Overhead: {overhead*100:.1f}%")
        
        return overhead > 0.1  # More than 10% slower

failure_analyzer = MixedPrecisionFailureModes()
pathological_failure = failure_analyzer.demonstrate_pathological_case()
small_model_overhead = failure_analyzer.benchmark_small_model_overhead()

if pathological_failure:
    print("\nWARNING: Mixed precision struggled with pathological parameter ranges")
if small_model_overhead:
    print("WARNING: Mixed precision added overhead to small model")