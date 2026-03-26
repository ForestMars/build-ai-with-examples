# Pipeline parallelism

In the [previous post](https://buildai.substack.com/p/tensor-parallelism-splitting-attention), we carved up attention heads like a distributed sushi chef in a Murakami story, splitting individual operations across GPUs, praying our gradients would synchronize properly, and like a Murakami protagonist, hoping their disparate worlds would somehow align in the end. Today, we're going vertical. Instead of splitting computations within layers, we're going to split the layers themselves across devices, creating a computational assembly line that would make Henry Ford weep with joy. Or at at least his digital twin.

Subscribe

Welcome to pipeline parallelism, where your x billion parameter transformer becomes a factory floor with each GPU responsible for a different stage of production. GPU 0 handles layers 1-8, GPU 1 takes layers 9-16, and so on. Simple, elegant, and absolutely riddled with the kind of subtle timing dependencies that will have you debugging phantom deadlocks at 3 AM while questioning your life choices. (I’m literally speaking from experience.)

The beautiful irony is that pipeline parallelism is simultaneously the most intuitive form of model parallelism (obviously you put different layers on different devices!) and the most diabolically complex to get right. It's like explaining how a car engine works versus actually building one that doesn't explode on the first try. (That joke will probably be lost on the next generation, since motors generally don’t explode.)

## The Assembly Line: Layers as Pipeline Stages

The core insight behind pipeline parallelism is deceptively simple. Your transformer has, say, 80 layers. You have 8 GPUs. Put 10 layers on each GPU, and boom, you've got yourself a pipeline. Data flows from GPU 0 through GPU 7, each device applying its layers in sequence.



📄 **[Code: pipelinestage.py](./pipelinestage.py)**



This naive implementation misses the entire point. We're still running everything sequentially on a single device. The magic happens when we actually distribute these stages across GPUs and start overlapping computation.

## The Bubble Problem: Why Pipeline Parallelism is Hard

Here's where pipeline parallelism gets interesting in all the wrong ways. Unlike data parallelism (where all GPUs work independently) or tensor parallelism (where GPUs collaborate on the same computation), pipeline parallelism creates dependencies. GPU 1 can't start working until GPU 0 finishes. GPU 2 waits for GPU 1. And so on.

This creates what we lovingly call "pipeline bubbles" which are periods where GPUs sit idle waiting for data from upstream stages. In the worst case, only one GPU is working at a time, giving you 1/8th the performance of a single device. Not exactly the scaling we were hoping for.

The solution is micro-batching: instead of sending one large batch through the pipeline, we split it into smaller micro-batches that can overlap in different stages.



📄 **[Code: microbatch.py](./microbatch.py)**



## GPipe vs PipeDream: The Scheduling Wars

There are two main approaches to pipeline scheduling, and choosing between them is like choosing between different types of controlled chaos.

**GPipe (Google's approach)**: Complete all forward passes before starting backward passes. Simple, synchronous, but creates large pipeline bubbles. Think of it as a batch processing system.

**PipeDream (Microsoft's approach)**: Interleave forward and backward passes to minimize bubbles. More complex, but much better hardware utilization. Think of it as a streaming system.

Those of you who know me, know I have a lot to say about batching vs streaming, as if it was a profound philosophical choice and not merely an architectural decision. Here's a simplified PipeDream implementation:



📄 **[Code: pipedreamstage.py](./pipedreamstage.py)**



This simplified `PipeDreamStage` code provides a high-level conceptual view of how interleaved forward and backward passes work. However, in a production environment, this approach has more hidden complexity than a smile in a Dostoyevsky novel. The `async` keywords are a hint at the real challenge: coordinating these concurrent tasks and managing weight versions without leading to deadlocks or incorrect gradients. While `torch.distributed.isend` and `irecv` provide non-blocking communication, using them directly in a naive way still requires manual management of the `requests` and `handles`, leading to complex race conditions and deadlocks. And you can't simply `await` them in a standard Python `asyncio` loop since `torch.distributed` operations are blocking by default.. The solution lies in a more robust scheduling and communication framework that handles these details for us, which we will explore next.

## Memory Management: The Hidden Complexity

Pipeline parallelism creates a subtle but critical memory management problem. During forward passes, you need to store activations for the backward pass. But with micro-batching, you might have multiple forward passes in flight before the first backward pass starts.

This means your memory usage grows linearly with the number of micro-batches. Too few micro-batches and you get pipeline bubbles. Too many and you run out of memory. Welcome to the optimization nightmare.



📄 **[Code: activationcheckpointing.py](./activationcheckpointing.py)**



## The Communication Pattern: Point-to-Point Chaos

Unlike the all-reduce patterns in data and tensor parallelism, pipeline parallelism uses point-to-point communication. GPU 0 only talks to GPU 1, GPU 1 talks to GPU 0 and GPU 2, and so on. This creates a beautiful linear communication pattern that scales perfectly... until it doesn't.

The problem is that point-to-point communication is much harder to optimize than collective operations. NCCL is heavily optimized for all-reduce, but peer-to-peer transfers can become bottlenecks, especially with high-bandwidth interconnects where the latency matters more than bandwidth.



📄 **[Code: optimizedp2pcommunication.py](./optimizedp2pcommunication.py)**



## Load Balancing: When Layers Aren't Created Equal

Here's a dirty secret about transformer layers: they're not all the same computational cost. Early layers often do more heavy lifting, attention patterns vary with depth, and some layers might have more parameters than others. If you naively split 80 layers into 8 groups of 10, you might end up with massive load imbalancing.



📄 **[Code: layerprofiler.py](./layerprofiler.py)**



## Putting It All Together: A Real Pipeline

Here's a complete pipeline parallelism implementation that actually works:



📄 **[Code: from.py](./from.py)**



As should be obvi, a production-ready pipeline parallel implementation is more complex than Nijinsky tryna choreograph Stravinsky while the orchestra is tuning up. It combines strategic model partitioning with sophisticated scheduling, memory management, and asynchronous communication. Each component, from the micro-batch scheduler to the activation checkpointer, is designed to chip away at the fundamental inefficiencies (pipeline bubbles and memory bottlenecks) that plague a naive approach. This architecture represents the culmination of a decade of research into distributed training, transforming a simple idea into a system that can train the world's largest AI models. And now you have the power.

## The Performance Reality Check

Ah but the power of pipeline parallelism is a double-edged sword. When it works well, you get excellent memory efficiency and can train models that would never fit on a single device. When it works poorly, you get pipeline bubbles that make your expensive GPU cluster perform *worse* than a single machine.

The efficiency formula is roughly:



📄 **[Code: snippet_7.py](./snippet_7.py)**



*(Cf. [Huang et al, 2018](https://arxiv.org/abs/1811.06965), [Narayanan et al, 2019](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf))*

With 8 stages and 16 micro-batches, you get about 70% efficiency. Not terrible, but not great either. The key is finding the sweet spot between memory usage and pipeline bubbles. We circle back on some advanced approaches later in the series.

## The Memory vs Bubble Time Trade-off

Here's the fundamental tension in pipeline parallelism: more micro-batches means better pipeline utilization but higher memory usage. Fewer micro-batches means lower memory usage but worse utilization. There's no free lunch (contrary to what we tell our new hires.) Sorry about that.



📄 **[Code: calculate_pipeline_efficiency.py](./calculate_pipeline_efficiency.py)**



## What's Next: The Hybrid Future

Pure pipeline parallelism is rarely the answer for modern large models. The real magic happens when you combine it with tensor parallelism and data parallelism in clever ways. Imagine splitting your model both horizontally (tensor parallelism within layers) and vertically (pipeline parallelism across layers), then replicating the whole thing across multiple data parallel groups. This is essentially data layout coherence for AI models, viz. maintaining alignment between your data organization and access patterns across the entire stack, from storage through compute kernels. Later we’ll explore how data type enforcement across the entire stack intersections with data layout coherence. The key insight is designing sharding strategies where horizontal and vertical data flows naturally align, creating end-to-end data locality optimization that maximizes bandwidth utilization efficiency at every level of the compute hierarchy.

But that's a story for another post (the last one in this unit) where we'll dive into 3D parallelism and learn how to orchestrate tensor, pipeline, and data parallelism simultaneously without losing our sanity.

For now, you have the tools to build production-ready pipeline parallel training. The code above will actually run and give you meaningful performance insights. Just don't blame me when you're debugging communication deadlocks at 3 AM while your GPUs burn electricity doing nothing. I’ll be busy stacking NVDA.

𒅃 The universal Turing machine processed one symbol at a time on an infinite tape. We've decided that's too slow, so we've built an assembly line of Turing machines, each processing different parts of the computation in parallel. Progress, or madness? It’s a real question. And honestly at this point, who is left to disagree that it’s probably both?

---

Next up: [Data Parallelism and Gradient Synchronization](https://buildai.substack.com/p/data-parallelism-and-gradient-synchronization) - Where we learn to love AllReduce as much as we hate communication bottlenecks.

*Questions about pipeline parallelism? Ready to share your own micro-batching horror stories? Paid subscribers drop them in the comments. Nothing builds engineering camaraderie like shared suffering over distributed training bugs. (Subscription is free, paid level has perks.)*

*← [prev: Tensor Parallelism](https://buildai.substack.com/p/tensor-parallelism-splitting-attention) | [next: Data Parallelism](https://buildai.substack.com/p/data-parallelism-and-gradient-synchronization) →*

*[BUILD AI](https://buildai.substack.com/p/build-ai-modules)*

Thanks for reading AI Engineering (with examples)! Subscribe for free to receive new posts and support my work.

Subscribe