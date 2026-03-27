import numpy as np
import torch

class PrecisionAnalyzer:
    """Analyze different floating point formats"""
    
    def __init__(self):
        self.formats = {
            'FP32': {'sign': 1, 'exponent': 8, 'mantissa': 23},
            'FP16': {'sign': 1, 'exponent': 5, 'mantissa': 10},
            'BF16': {'sign': 1, 'exponent': 8, 'mantissa': 7},
        }
        
    def analyze_range_and_precision(self):
        """Compare numerical properties of different formats"""
        
        print("Floating Point Format Analysis:")
        print("-" * 50)
        
        for format_name, bits in self.formats.items():
            # Calculate range
            max_exponent = 2**(bits['exponent'] - 1) - 1
            min_exponent = -max_exponent + 1
            
            largest_normal = 2**max_exponent * (2 - 2**(-bits['mantissa']))
            smallest_normal = 2**min_exponent
            
            # Calculate precision (machine epsilon)
            machine_eps = 2**(-bits['mantissa'])
            
            print(f"\n{format_name}:")
            print(f"  Largest normal: {largest_normal:.2e}")
            print(f"  Smallest normal: {smallest_normal:.2e}")
            print(f"  Machine epsilon: {machine_eps:.2e}")
            print(f"  Dynamic range: {np.log10(largest_normal/smallest_normal):.1f} orders of magnitude")
            
    def gradient_underflow_demo(self):
        """Demonstrate gradient underflow in different precisions"""
        
        # Simulate tiny gradients that might underflow
        gradients = torch.tensor([1e-4, 1e-5, 1e-6, 1e-7, 1e-8], dtype=torch.float32)
        
        print("\nGradient Underflow Analysis:")
        print("-" * 30)
        
        for dtype, name in [(torch.float32, 'FP32'), (torch.float16, 'FP16'), (torch.bfloat16, 'BF16')]:
            converted = gradients.to(dtype)
            underflowed = (converted == 0).sum().item()
            
            print(f"{name}: {underflowed}/5 gradients underflowed to zero")
            print(f"  Values: {converted.tolist()}")

analyzer = PrecisionAnalyzer()
analyzer.analyze_range_and_precision()
analyzer.gradient_underflow_demo()