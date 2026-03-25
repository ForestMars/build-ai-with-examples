# Enable detailed distributed logging
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Check for synchronization issues
def debug_sync_point(name):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} reached {name}")
        dist.barrier()  # Force synchronization
        print(f"Rank {dist.get_rank()} passed {name}")

# Monitor gradient norms across ranks
def check_gradient_sync(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Rank {dist.get_rank()}, {name}: grad_norm = {grad_norm}")

