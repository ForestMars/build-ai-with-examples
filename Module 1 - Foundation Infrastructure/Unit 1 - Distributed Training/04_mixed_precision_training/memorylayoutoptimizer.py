class MemoryLayoutOptimizer:
    """Optimize memory layouts for mixed precision training"""
    
    def __init__(self):
        self.device = torch.cuda.current_device()
        
    def benchmark_tensor_layouts(self, shapes, dtypes):
        """Benchmark different tensor storage patterns"""
        
        results = {}
        
        for shape in shapes:
            for dtype in dtypes:
                # Contiguous layout
                tensor_c = torch.randn(shape, dtype=dtype, device='cuda')
                
                # Strided layout (simulating transpose or permute)
                tensor_s = tensor_c.transpose(-1, -2)
                
                # Benchmark matrix multiplication
                other = torch.randn(shape[-1], 512, dtype=dtype, device='cuda')
                
                # Warmup
                for _ in range(10):
                    _ = torch.matmul(tensor_c, other)
                    _ = torch.matmul(tensor_s, other)
                torch.cuda.synchronize()
                
                # Benchmark contiguous
                start_time = time.perf_counter()
                for _ in range(100):
                    result_c = torch.matmul(tensor_c, other)
                torch.cuda.synchronize()
                time_contiguous = time.perf_counter() - start_time
                
                # Benchmark strided
                start_time = time.perf_counter()
                for _ in range(100):
                    result_s = torch.matmul(tensor_s, other)
                torch.cuda.synchronize()
                time_strided = time.perf_counter() - start_time
                
                key = f"{shape}_{dtype}"
                results[key] = {
                    'contiguous_time': time_contiguous,
                    'strided_time': time_strided,
                    'slowdown': time_strided / time_contiguous
                }
                
        return results
    
    def analyze_tensor_core_utilization(self):
        """Analyze tensor core utilization for different shapes/types"""
        
        # Tensor cores work best with specific shape alignments
        shapes = [
            (16, 64, 64),    # Good alignment
            (17, 65, 65),    # Poor alignment
            (32, 128, 128),  # Excellent alignment
            (31, 127, 127),  # Poor alignment
        ]
        
        dtypes = [torch.float16, torch.bfloat16]
        
        print("Tensor Core Utilization Analysis:")
        print("-" * 50)
        
        for shape in shapes:
            for dtype in dtypes:
                A = torch.randn(shape[0], shape[1], dtype=dtype, device='cuda')
                B = torch.randn(shape[1], shape[2], dtype=dtype, device='cuda')
                
                # Warmup
                for _ in range(10):
                    _ = torch.matmul(A, B)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(1000):
                    result = torch.matmul(A, B)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                
                # Calculate theoretical FLOPS
                flops = 2 * shape[0] * shape[1] * shape[2] * 1000  # 1000 iterations
                achieved_tflops = flops / elapsed / 1e12
                
                dtype_name = str(dtype).split('.')[-1]
                print(f"Shape {shape}, {dtype_name}: {achieved_tflops:.1f} TFLOPS")
                
        print("\nNote: Shapes aligned to multiples of 8/16 typically achieve higher TFLOPS")
        print("due to better tensor core utilization.")

class MixedPrecisionMemoryManager:
    """Advanced memory management for mixed precision training"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fp32_parameters = []
        self.fp16_parameters = []
        self.parameter_groups = {}
        
        self.analyze_parameter_precision_requirements()
        
    def analyze_parameter_precision_requirements(self):
        """Categorize parameters by precision requirements"""
        
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['norm', 'bias']):
                # Keep normalization layers and biases in FP32
                self.fp32_parameters.append((name, param))
            else:
                # Main weights can use FP16
                self.fp16_parameters.append((name, param))
                
        print(f"Parameter Precision Analysis:")
        print(f"  FP32 parameters: {len(self.fp32_parameters)}")
        print(f"  FP16 parameters: {len(self.fp16_parameters)}")
        
    def create_parameter_groups(self):
        """Create optimizer parameter groups with different precisions"""
        
        fp32_group = {
            'params': [param for _, param in self.fp32_parameters],
            'lr': 1e-4,
            'precision': 'fp32'
        }
        
        fp16_group = {
            'params': [param for _, param in self.fp16_parameters],
            'lr': 1e-4,
            'precision': 'fp16'
        }
        
        return [fp32_group, fp16_group]
    
    def convert_model_precision(self):
        """Convert model to mixed precision"""
        
        for name, param in self.fp16_parameters:
            # Convert to FP16 in-place
            param.data = param.data.half()
            
        print("Model converted to mixed precision")
        
    def get_memory_stats(self):
        """Calculate memory usage statistics"""
        
        fp32_memory = sum(param.numel() * 4 for _, param in self.fp32_parameters)
        fp16_memory = sum(param.numel() * 2 for _, param in self.fp16_parameters)
        
        total_memory = fp32_memory + fp16_memory
        pure_fp32_memory = sum(param.numel() * 4 for _, param in self.model.named_parameters())
        
        savings = (pure_fp32_memory - total_memory) / pure_fp32_memory
        
        return {
            'fp32_memory_gb': fp32_memory / 1e9,
            'fp16_memory_gb': fp16_memory / 1e9,
            'total_memory_gb': total_memory / 1e9,
            'memory_savings': savings,
            'savings_gb': (pure_fp32_memory - total_memory) / 1e9
        }

# Demo
optimizer = MemoryLayoutOptimizer()
results = optimizer.benchmark_tensor_layouts(
    shapes=[(128, 512), (256, 1024)], 
    dtypes=[torch.float16, torch.float32]
)

for key, result in results.items():
    print(f"{key}: Strided is {result['slowdown']:.2f}x slower than contiguous")