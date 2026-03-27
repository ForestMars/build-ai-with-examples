import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    enabled: bool = True
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_grad_norm: float = 1.0
    
    # Advanced options
    use_bfloat16: bool = False  # Use BF16 instead of FP16
    keep_batchnorm_fp32: bool = True
    loss_scale_window: int = 1000
    min_loss_scale: float = 1.0

class ProductionMixedPrecisionTrainer:
    """Production-ready mixed precision training with all optimizations"""
    
    def __init__(self, model: nn.Module, config: MixedPrecisionConfig):
        self.model = model
        self.config = config
        
        # Setup precision management
        if config.enabled:
            self.setup_mixed_precision()
            
        # Enhanced gradient scaler
        self.scaler = GradScaler(
            init_scale=config.init_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval
        )
        
        # Training statistics
        self.step_count = 0
        self.overflow_count = 0
        self.scale_history = []
        self.loss_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def setup_mixed_precision(self):
        """Configure model for mixed precision training"""
        
        if self.config.use_bfloat16:
            # Convert appropriate layers to BF16
            self.convert_to_bfloat16()
        
        # Keep batch normalization layers in FP32 for stability
        if self.config.keep_batchnorm_fp32:
            self.keep_batchnorm_fp32()
            
        self.logger.info(f"Mixed precision setup complete. BF16: {self.config.use_bfloat16}")
        
    def convert_to_bfloat16(self):
        """Convert model to BF16 where appropriate"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
                    
    def keep_batchnorm_fp32(self):
        """Ensure batch normalization stays in FP32"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.float()  # Keep in FP32
                
    def training_step(self, batch, targets, optimizer) -> Dict:
        """Complete training step with mixed precision"""
        
        step_start_time = time.perf_counter()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast
        if self.config.enabled:
            with autocast(dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float16):
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, targets)
        else:
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, targets)
            
        # Backward pass
        if self.config.enabled:
            # Scale loss to prevent gradient underflow
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            ).item()
            
            # Check for gradient overflow
            overflow = self.check_overflow()
            
            if not overflow:
                # Safe to take optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                self.overflow_count += 1
                self.logger.warning(f"Gradient overflow at step {self.step_count}")
                
        else:
            # Standard FP32 training
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            ).item()
            optimizer.step()
            overflow = False
            
        # Update statistics
        self.step_count += 1
        current_scale = self.scaler.get_scale() if self.config.enabled else 1.0
        self.scale_history.append(current_scale)
        self.loss_history.append(loss.item())
        
        step_time = time.perf_counter() - step_start_time
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'scale': current_scale,
            'overflow': overflow,
            'step_time': step_time
        }
        
    def compute_loss(self, outputs, targets):
        """Compute loss - always in FP32 for numerical stability"""
        return nn.CrossEntropyLoss()(outputs.float(), targets)
        
    def check_overflow(self) -> bool:
        """Check if gradients have overflowed"""
        
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return True
        return False
        
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        
        if not self.loss_history:
            return {}
            
        overflow_rate = self.overflow_count / max(self.step_count, 1)
        avg_loss = sum(self.loss_history[-100:]) / min(len(self.loss_history), 100)
        current_scale = self.scale_history[-1] if self.scale_history else 1.0
        
        return {
            'total_steps': self.step_count,
            'overflow_count': self.overflow_count,
            'overflow_rate': overflow_rate,
            'current_scale': current_scale,
            'avg_recent_loss': avg_loss,
            'scale_stability': self.analyze_scale_stability()
        }
        
    def analyze_scale_stability(self) -> Dict:
        """Analyze gradient scaling stability"""
        
        if len(self.scale_history) < 100:
            return {'status': 'insufficient_data'}
            
        recent_scales = self.scale_history[-100:]
        scale_changes = sum(1 for i in range(1, len(recent_scales)) 
                          if recent_scales[i] != recent_scales[i-1])
        
        stability_score = 1.0 - (scale_changes / 100)
        
        return {
            'status': 'stable' if stability_score > 0.9 else 'unstable',
            'stability_score': stability_score,
            'scale_changes_per_100_steps': scale_changes,
            'current_scale': recent_scales[-1],
            'min_scale': min(recent_scales),
            'max_scale': max(recent_scales)
        }
        
    def save_checkpoint(self, filepath: str, optimizer, epoch: int):
        """Save training checkpoint with mixed precision state"""
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.get_training_stats()
        }
        
        if self.config.enabled:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
        
    def load_checkpoint(self, filepath: str, optimizer):
        """Load training checkpoint"""
        
        checkpoint = torch.load(filepath, map_location='cuda')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step_count = checkpoint.get('step', 0)
        
        if self.config.enabled and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.logger.info(f"Checkpoint loaded: {filepath}, step {self.step_count}")
        
        return checkpoint.get('epoch', 0)

# Example usage with monitoring
def train_with_mixed_precision_monitoring():
    """Example training loop with comprehensive monitoring"""
    
    # Setup
    model = SimpleTransformerLayer().cuda()
    config = MixedPrecisionConfig(
        enabled=True,
        use_bfloat16=False,  # Use FP16
        init_scale=32768.0,
        max_grad_norm=1.0
    )
    
    trainer = ProductionMixedPrecisionTrainer(model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(1000):
        # Generate synthetic batch
        batch = torch.randn(16, 256, 1024).cuda()
        targets = torch.randint(0, 1000, (16,)).cuda()
        
        # Training step
        step_stats = trainer.training_step(batch, targets, optimizer)
        
        # Periodic monitoring
        if step % 100 == 0:
            training_stats = trainer.get_training_stats()
            
            print(f"\nStep {step}:")
            print(f"  Loss: {step_stats['loss']:.4f}")
            print(f"  Grad Norm: {step_stats['grad_norm']:.4f}")
            print(f"  Scale: {step_stats['scale']:.0f}")
            print(f"  Overflow Rate: {training_stats['overflow_rate']:.3f}")
            
            # Scale stability analysis
            scale_analysis = training_stats.get('scale_stability', {})
            if scale_analysis.get('status') == 'unstable':
                print(f"  WARNING: Gradient scaling is unstable!")
                print(f"    Scale changes: {scale_analysis['scale_changes_per_100_steps']}")
                
        # Save checkpoint periodically
        if step % 500 == 0 and step > 0:
            trainer.save_checkpoint(f'checkpoint_step_{step}.pt', optimizer, step // 100)

# Run the demo (uncomment to execute)
# train_with_mixed_precision_monitoring()