class GradientScalingAnalysis:
    """Deep dive into gradient scaling strategies"""
    
    def __init__(self):
        self.scale_values = [2**i for i in range(1, 16)]  # Powers of 2 from 2 to 32768
        
    def analyze_gradient_distribution(self, model: nn.Module, data_loader, num_batches=10):
        """Analyze the distribution of gradient magnitudes"""
        
        gradient_norms = []
        
        # Collect gradient statistics
        for batch_idx, (batch, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            # Forward pass
            output = model(batch.cuda())
            loss = nn.CrossEntropyLoss()(output, targets.cuda())
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Collect gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms.append((name, grad_norm))
                    
        # Analysis
        all_norms = [norm for _, norm in gradient_norms]
        min_norm = min(all_norms)
        max_norm = max(all_norms)
        median_norm = sorted(all_norms)[len(all_norms)//2]
        
        print(f"Gradient Analysis:")
        print(f"  Min gradient norm: {min_norm:.2e}")
        print(f"  Median gradient norm: {median_norm:.2e}")  
        print(f"  Max gradient norm: {max_norm:.2e}")
        
        # Check FP16 representability
        fp16_min = 2**(-14)  # Smallest FP16 subnormal
        underflow_count = sum(1 for norm in all_norms if norm < fp16_min)
        print(f"  Gradients that would underflow in FP16: {underflow_count}/{len(all_norms)} ({underflow_count/len(all_norms)*100:.1f}%)")
        
        return min_norm, max_norm, median_norm
        
    def optimal_scale_search(self, model: nn.Module, data_loader):
        """Find optimal gradient scaling factor"""
        
        min_norm, max_norm, median_norm = self.analyze_gradient_distribution(model, data_loader)
        
        # We want to scale gradients so the median is well within FP16 range
        # But not so much that large gradients overflow
        fp16_max = 65504  # Max FP16 value
        
        # Conservative approach: scale so median gradient is around 1e-3
        target_median = 1e-3
        suggested_scale = target_median / median_norm
        
        # But make sure we don't overflow the largest gradients
        max_safe_scale = fp16_max / (max_norm * 2)  # Factor of 2 for safety
        
        optimal_scale = min(suggested_scale, max_safe_scale)
        
        # Round to nearest power of 2 for efficiency
        optimal_scale = 2 ** round(np.log2(optimal_scale))
        
        print(f"\nOptimal Scale Analysis:")
        print(f"  Target median gradient: {target_median:.2e}")
        print(f"  Suggested scale: {suggested_scale:.0f}")
        print(f"  Max safe scale: {max_safe_scale:.0f}")
        print(f"  Optimal scale (power of 2): {optimal_scale:.0f}")
        
        return optimal_scale

class AdaptiveGradScaler:
    """More sophisticated gradient scaler that adapts during training"""
    
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        
        # Tracking variables
        self.growth_tracker = 0
        self.inf_count = 0
        self.total_steps = 0
        
        print(f"Initialized AdaptiveGradScaler with scale={init_scale}")
        
    def scale_gradients(self, loss):
        """Scale loss for backward pass"""
        return loss * self.scale
        
    def unscale_gradients(self, optimizer):
        """Unscale gradients and check for infs/nans"""
        
        has_inf = False
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.div_(self.scale)
                    
                    # Check for inf/nan
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        has_inf = True
                        break
            if has_inf:
                break
                
        return has_inf
        
    def update_scale(self, has_inf):
        """Update scaling factor based on whether we found infs"""
        
        self.total_steps += 1
        
        if has_inf:
            # Overflow detected, reduce scale
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
            self.inf_count += 1
            print(f"Step {self.total_steps}: Overflow detected, reducing scale to {self.scale:.0f}")
        else:
            # No overflow, consider increasing scale
            self.growth_tracker += 1
            
            if self.growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_tracker = 0
                print(f"Step {self.total_steps}: Increasing scale to {self.scale:.0f}")
                
    def get_stats(self):
        """Get training statistics"""
        overflow_rate = self.inf_count / max(self.total_steps, 1)
        return {
            'current_scale': self.scale,
            'total_steps': self.total_steps,
            'overflow_count': self.inf_count,
            'overflow_rate': overflow_rate
        }

# Example usage in training loop
def training_loop_with_adaptive_scaling():
    """Example of training with adaptive gradient scaling"""
    
    model = SimpleTransformerLayer().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = AdaptiveGradScaler(init_scale=32768.0)
    
    # Synthetic training data
    batch_size, seq_len, hidden_dim = 16, 256, 1024
    
    for step in range(1000):
        # Generate batch
        batch = torch.randn(batch_size, seq_len, hidden_dim).cuda()
        targets = torch.randint(0, 1000, (batch_size,)).cuda()
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(batch)
            loss = nn.CrossEntropyLoss()(output, targets)
            
        # Scale loss and backward pass
        scaled_loss = scaler.scale_gradients(loss)
        scaled_loss.backward()
        
        # Unscale and check for overflows
        has_inf = scaler.unscale_gradients(optimizer)
        
        if not has_inf:
            # Safe to take optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Update scaling factor
        scaler.update_scale(has_inf)
        
        if step % 100 == 0:
            stats = scaler.get_stats()
            print(f"Step {step}: Loss={loss.item():.4f}, Scale={stats['current_scale']:.0f}, Overflow Rate={stats['overflow_rate']:.3f}")

# Run the demo
# training_loop_with_adaptive_scaling()