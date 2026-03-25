# Tensor parallelism

In 1936, Alan Turing proved that any computation could be reduced to a single machine shuttling symbols on an endless tape. In 2025, we've decided that's not nearly fast enough. So we're going to take those computations, chop them up with the precision of a sushi master, and scatter the pieces across multiple GPUs while praying to the NCCL gods that our gradients synchronize properly.

Welcome to our first installment, tensor parallelism, the art of splitting individual operations across devices rather than entire layers or data batches. Where pipeline parallelism (which we’ll cover in part 2) says "let's put different layers on different devices," and data parallelism (the subject of part 3) says "let's run the same model on different data," tensor parallelism boldly declares "let's put *parts of the same layer* on different devices and see what happens."

What happens, as it turns out, is both beautiful and terrifying. Beautiful because you can train models that would never fit on a single GPU. Terrifying because you're now debugging synchronization bugs across dozens of CUDA streams while your gradients sometimes mysteriously vanish into the void. Which we’ll deal with in further installments. One step at a time and all that.

## The Fundamental Challenge: When Models Don't Fit

Let's start with a harsh reality check. A 70B parameter model in FP16 precision requires roughly 140GB of memory just to store the weights. Add gradients, optimizer states, and activations, and you're looking at 400GB+ of memory. The largest single GPU you can buy today (H100 with 80GB HBM) isn't even close.

This is where tensor parallelism becomes not just useful, but indispensable. Instead of saying "this layer is too big for one GPU," we say "this matrix multiplication is too big for one GPU, and so let's split the matrices themselves."

## Multi-Head Attention: The Perfect Victim

Multi-head attention is almost criminally well-suited for tensor parallelism. Consider the canonical transformer attention mechanism:



📄 **[Code: naivemultiheadattention.py](./naivemultiheadattention.py)**



Notice the key insight: those attention heads are embarrassingly parallel. Head 0 doesn't need to know what Head 7 is doing until the very end when we concatenate and project through `W_o`. This natural structure makes attention layers perfect candidates for tensor parallelism. In actuality it was about 2 years between Attention is All You Need, and Megatron, the first real world implementation of tensor parallelism, legendarily born out of a late-night, caffeine-fueled aha moment.

## The Art of the Split: Column vs Row Parallelism

Tensor parallelism comes in two main flavors, and understanding the difference might save you from me making too many jokes about mysterious deadlocks at 3 AM. Or it might not. But this is where we see the salience of data layout coherence start to make sense.

**Column Parallelism**: Split the output dimension. Each GPU gets a subset of the columns. **Row Parallelism**: Split the input dimension. Each GPU gets a subset of the rows.

The key is knowing when to use which approach. Those painful 3 AM sessions are very often the result of someone picking the wrong parallelism strategy for their specific layer, or worse, didn’t bother to think through how the splits compose across the full network. I’m not naming names here.   
  
For multi-head attention, we use column parallelism on the Q, K, V projections (each GPU handles a subset of heads), and row parallelism on the output projection (each GPU contributes to the final result.)



📄 **[Code: tensorparallelmultiheadattention.py](./tensorparallelmultiheadattention.py)**



## The Communication Tax: When GPUs Talk to Each Other

That `all_reduce` call (3rd spelling if you’re keeping score) is where the rubber meets the road as we say in meetings. It's also where most of your performance goes to die if you're not careful.

AllReduce is a collective operation that combines values from all processes. In our case, each GPU has computed a partial result for the output projection, and we need to sum them all together. This happens in two phases:

1. **Reduce-Scatter**: Each GPU gets the sum for a subset of elements
2. **AllGather**: Each GPU broadcasts its subset to all other GPUs

The naive implementation above has a problem: we're doing an AllReduce on every forward pass. For a model with dozens of attention layers, this becomes a communication nightmare. The solution is to be smarter about when and where we synchronize.

### The Interconnect Mandate: Why Topology Matters

Communication between replicas in the same tensor parallel group occurs much more frequently than in [data parallel](https://buildai.substack.com/p/data-parallelism-and-gradient-synchronization) groups or between [pipeline stages](https://buildai.substack.com/p/pipeline-parallelism-forwardbackward). In TP, you aren’t just syncing at the end of a pass; you are syncing multiple times within every single layer. Because the time spent communicating increases drastically relative to computation, the interconnect tax is highest here. Replicas in the same tensor parallel group **must** be placed on the highest bandwidth, lowest latency fabric available.

Historically, this meant keeping TP groups within a single node via NVLink. However, with the release of macOS Tahoe, the playfield has expanded. Apple’s soon-to-be-released implementation of UMA over RDMA allows for memory-to-memory transfers across nodes that bypass the CPU bottleneck. While NVLink remains the gold standard for raw throughput, Tahoe's RDMA integration means we can now consider stretched tensor parallel groups across Apple Silicon clusters, provided the network fabric supports the near-zero-copy requirements of the Tahoe 26.x stack.

## Memory Layout: The Devil in the Details

Here's something the textbooks don't tell you: tensor parallelism is as much about memory layout as it is about computation. Get the layout wrong, and your GPUs will spend more time shuffling data than doing math. Overhead is like the moon that way. As a colleague recently put it, it’s about seeing the boundary conditions up close and personal.



📄 **[Code: optimizedtensorparallelattention.py](./optimizedtensorparallelattention.py)**



## The Gradient Flow: Backward Pass Complications

Forward pass is just the beginning. The real complexity emerges during backpropagation, where gradients need to flow backward through the distributed computation graph. The key insight is that communication operations need to be differentiable. PyTorch's distributed package handles this for you, but understanding what's happening under the hood is crucial for debugging. This is not something Claude Code is going to fix for you while you run for coffee.   
  
The subtlety here isn’t in the math since `matmul` and `add` are, as any PyTorch user knows, already differentiable, but rather in the communication across devices. Once you start slicing matrices across GPUs, you need collective operations (`all_reduce`, `all_gather`, `scatter`) to stitch partial results back together. PyTorch’s autograd engine can only propagate gradients through these collectives **if** you use the `torch.distributed` versions that are registered with backward rules. That’s literally what makes a tensor-parallel linear layer special: it’s not just splitting weights, it’s making sure the communication ops that glue the shards back together remain a first class participant in your computation graph, so gradients flow all the way back to every GPU’s parameters. Without that, you’d get the dreaded vanishing gradients which would emerge not from the model design but from your parallelism strategy itself.



📄 **[Code: tensorparallellinear.py](./tensorparallellinear.py)**



## Putting It All Together: A Complete Example

Let's build a simple but complete tensor parallel transformer layer that you can actually run:



📄 **[Code: tensorparallellinear_1.py](./tensorparallellinear_1.py)**



## Performance Considerations: The Good, The Bad, The Ugly

**The Good**: Tensor parallelism can give you near-linear speedup for the forward pass when your model is compute-bound and your interconnect is fast (NVLink, InfiniBand).

**The Bad**: Communication overhead can dominate for smaller models or slower interconnects. Every AllReduce is a synchronization point that can create bubbles in your pipeline.

**The Ugly**: Memory overhead from storing partial results, complexity of debugging distributed failures, and the fact that your model architecture now depends on your hardware configuration.

## Debugging Distributed Nightmares

When (not if) your tensor parallel training hangs or produces NaNs, here's your debugging toolkit:



📄 **[Code: debug_sync_point.py](./debug_sync_point.py)**



`debug_sync_point(tag)` is most useful as a way of confirming that all ranks are progressing in lockstep through the forward and backward passes. Since it inserts a barrier and prints a tag, you can drop it before and after key collectives like `all_reduce`, `all_gather`, or `reduce_scatter`. If training hangs and one rank never prints the tag, you immediately know the others are stuck waiting at that sync point, which almost always indicates a shape mismatch or diverging control flow. By sprinkling distinct tags through the attention module, you get a breadcrumb trail that shows exactly where ranks fall out of sync.

`check_gradient_sync(param)` is a complementary tool for verifying that gradient communication is doing what you think it is. After calling `loss.backward()`, you can gather the gradients for a parameter across all ranks and compare them. If every rank shows the same gradient, you’ve accidentally replicated the parameter instead of sharding it. Whoopsie. If the gradients are completely different with no overlap, your slicing logic or head assignment is off. The healthy case is when each shard has gradients that make sense locally and combine to match the expected full gradient. A quick way to spot issues is to print the L2 norm of the local gradient alongside the gathered one; if the magnitudes diverge wildly, something is wrong in your sync logic.

## The Road Ahead

Tensor parallelism is just the beginning. In the next post, we'll dive into pipeline parallelism, where we'll split models across layers instead of within layers, introducing the delightful complexity of micro-batching and pipeline bubbles.

But for now, you have the tools to split attention heads across GPUs like a distributed computing samurai. The code above will actually run (assuming you have multiple GPUs and your NCCL installation isn't cursed by the networking gods). Paid subscribers have access to all code in the [Build AI with Examples Github repository](https://github.com/ForestMars/build-ai-with-examples). (otherwise 404, bc that’s how Github rolls 🙄)

𒅃 The universal Turing machine could simulate any computation on a single tape. We've decided that's not fast enough, so we're simulating our computations across multiple tapes simultaneously. Turing would either be proud or deeply confused. Probably a little of both.

---

Next up: [Pipeline Parallelism](https://buildai.substack.com/p/pipeline-parallelism-forwardbackward): Where we split models by layers and learn to love micro-batching bubbles.

*Have questions about tensor parallelism? Want to share your own distributed training war stories? Paid subscribers can drop them in the comments. Misery loves company, especially when that misery involves CUDA out-of-memory errors at 3 AM. Which is only midnight in SF, so the real suffering hasn't even started yet.*

*[BUILD AI](https://buildai.substack.com/p/build-ai-modules)*

Thanks for reading AI Engineering (with examples)! Subscribe for free to receive new posts and support my work.

Subscribe