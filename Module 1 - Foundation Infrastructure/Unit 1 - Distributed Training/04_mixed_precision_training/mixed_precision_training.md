# Mixed precision training

In 1985, the IEEE 754 standard declared that 32 bits should be enough for anyone to represent a floating point number. For the next three decades, we dutifully stored every weight, every gradient, every activation in glorious FP32 precision, as if the universe itself demanded that level of numerical fidelity. Then, much like IBM’s PC/AT smashed the arbitrary barrier of the 5150, someone clever at NVIDIA did the math and realized we were basically using a Ferrari to deliver pizza. Which honestly sounds like the perfect job for crypto-enthusiasts, but we over here doing AI.

Welcome to mixed precision training, where we've discovered that most of our neural networks are surprisingly tolerant of numerical approximation, and cutting our memory usage in half is worth the occasional gradient that vanishes into the numerical void. It’s almost like a ACID vs. BASE redux.

The beautiful irony is that mixed precision training represents a kind of computational maturity we've been slowly approaching for decades. Just as Turing showed us that infinite computational power could emerge from the simplest operations, and the Internet showed how reliable message delivery can be guaranteed by an unreliable underlying transport, mixed precision shows us that acceptable intelligence can emerge from deliberately imprecise computation. We're trading numerical purity for practical scale, and it turns out the neural networks don't actually care as much as we thought they would. Interesting how these breakthrough moments in computing consistently involve letting go of perfectionist assumptions we didn't even realize we were making.

This is the post where we learn to love approximation, embrace numerical instability, and somehow make our models train faster while using half the memory. It's like discovering you can get to work faster by taking the scenic route instead of the highway: counterintuitive, slightly risky, but surprisingly effective when you realize the highway was always congested with unnecessary precision. Just don’t go telling everyone.

## The Precision Paradox: When Less is More

Let's start with a reality check about what we're actually doing when we train neural networks. We're essentially performing millions of tiny numerical updates, hoping that the accumulated effect of all these micro-adjustments will somehow converge to something intelligent. Or failing that, something viral. The question is: do we really need 32 bits of precision for each of these adjustments?

The answer, as it turns out, is a resounding "mostly no." Neural networks exhibit a remarkable property called **precision resilience**: they can tolerate significant numerical approximation without losing their ability to learn useful representations. I would say that’s not just convenient, it's philosophically profound, except that philosophy has always been a kind of intellectual convenience. The intelligence that emerges from these systems doesn't require perfect numerical precision. It's robust to the kind of approximation that makes engineers uncomfortable but makes GPUs very, very happy. And Jensen even happier.



📄 **[Code: precisioncomparison.py](./precisioncomparison.py)**



The numbers are compelling. FP16 gives us roughly 2x memory savings for both parameters and activations. But here's where it gets interesting: the speedup isn't just from using less memory. Modern GPUs have specialized tensor cores (introduced in Volta) that can perform FP16 operations significantly faster than FP32. The H100's tensor cores can deliver up to 989 TOPS for FP16, compared to 67 TFLOPS for FP32. That's not a typo, we're talking about an order of magnitude difference in peak theoretical performance. YMMV.

## The Numerical Precision Hierarchy: FP32, FP16, BF16, and Beyond

Not all reduced precision formats are created equal. The choice between FP16, BF16, and other formats involves understanding the fundamental tradeoffs between numerical range, precision, and hardware support.



📄 **[Code: precisionanalyzer.py](./precisionanalyzer.py)**



Holy Toledo! Talk about trust but verify, that’s a 66 order of magnitude difference, empirically tested. The key insight here is that FP16 and BF16 make different tradeoffs. FP16 has higher precision (10 mantissa bits vs 7) but smaller range (5 exponent bits vs 8) as it doesn’t sacrifice the mantissa at the altar of higher exponentiality, as BF16 does, and thus yields higher precision, but at the cost of those measured 66 orders of magnitude in dynamic range. (I ran this on a MacBook Pro M2.) BF16 has the same range as FP32 but lower precision. The key is that for neural network training, range often matters more than precision; we'd rather represent a tiny gradient approximately than lose it entirely to underflow.

This is why Google chose BF16 for TPUs and why it's becoming increasingly popular for training large models. The extended range means fewer gradients vanish into numerical zero, which is critical for training stability.

## Automatic Mixed Precision: The PyTorch AutoCast Magic

PyTorch's Automatic Mixed Precision (AMP) is basically a sophisticated type system for floating point operations. It automatically chooses the appropriate precision for each operation based on numerical stability requirements. Matrix multiplications get FP16 for speed, while loss computations stay in FP32 for accuracy. If only we had this for neural nets.



📄 **[Code: mixedprecisiontrainer.py](./mixedprecisiontrainer.py)**



What's happening under the hood is quite sophisticated. The `autocast` context manager maintains a whitelist of operations that are safe to run in FP16 (like matrix multiplications) and a blacklist of operations that should stay in FP32 (like loss computations and batch normalization). This automatic type promotion ensures numerical stability while maximizing the performance benefits of reduced precision.

## Gradient Scaling: Preventing the Vanishing Gradient Apocalypse

Here's where mixed precision training gets philosophically interesting. Yes, I always find a way to make \*everything\* philosophically interesting, but in this case it’s even more true. The fundamental problem is that gradients in deep networks are often very small (like much smaller than the smallest representable FP16 number.) In FP32, a gradient of 1e-7 is perfectly fine. In FP16, its precision notwithstanding, it underflows to zero, and your model stops learning. Whoopsie.

The solution is gradient scaling: we multiply the loss by a large constant before computing gradients, then scale the gradients back down before the optimizer step. It's like temporarily amplifying a quiet signal to prevent it from getting lost in the noise.



📄 **[Code: gradientscalinganalysis.py](./gradientscalinganalysis.py)**



The beauty of adaptive gradient scaling is that it's essentially a feedback control system for numerical precision. Very cybernetic. When gradients overflow (indicating our scale is too high), we back off. When we go many steps without overflow (indicating we could safely scale higher), we increase the scale to preserve more gradient information. The kind of math Norbert Weiner did in his head. For fun.

## Memory Layout Optimization: CUDA Shoulda Woulda

Mixed precision isn't just about choosing FP16 vs FP32, it's about optimizing memory layouts to maximize GPU throughput. Modern GPUs have complex memory hierarchies, and the way you store and access mixed-precision data can dramatically impact performance. This is not something you’re likely to get right or even come close using a coding assistant, and in any case you’re going to need to know how this works under the hood. High-level abstractions are undeniably helpful but not sufficient for serious distributed systems work.



📄 **[Code: memorylayoutoptimizer.py](./memorylayoutoptimizer.py)**



The key insight here is that mixed precision extends far beyond numerical precision to encompass how modern GPU architectures actually execute computations. Tensor cores have specific alignment requirements, and violating them can cost you significant performance even when using the "right" data types. (More on types later. Much more, in fact.)

## Production Mixed Precision: Putting It All Together

Let's build a complete mixed precision training system that handles all the edge cases and optimizations we've discussed:



📄 **[Code: class.py](./class.py)**



## The Philosophy of Approximate Computation

There's something deeply fascinating about mixed precision training from a computational philosophy perspective. We're essentially embracing the idea that intelligence doesn't require perfect precision but that the emergent properties of neural networks are robust to the kind of numerical approximation that would that would violate everything traditional numerical analysts hold sacrosanct about computational accuracy.

This connects back to the broader themes we've been exploring in this series. Just as Turing showed us that universal computation could emerge from the simplest operations, along with the other examples of building reliable systems using unreliable primitives, mixed precision shows us that effective learning can emerge from deliberately imprecise computation. The networks don't need every gradient to be represented with perfect fidelity; they need enough signal to navigate the loss landscape effectively.

It's almost like discovering that you don't need a microscope to see the big picture (wait, what?) Sometimes a slightly blurry view is not only sufficient but actually more efficient for the task at hand.

## Performance Reality Check: When Mixed Precision Fails

But let's be real for a moment. Mixed precision isn't magic, and it doesn't always work. There are entire classes of problems where the numerical precision requirements are genuinely stringent, where gradient scaling becomes a constant battle, and where the memory savings aren't worth the engineering complexity.



📄 **[Code: mixedprecisionfailuremodes.py](./mixedprecisionfailuremodes.py)**



## The Road Ahead: Beyond FP16

As we wrap up our deep dive into mixed precision training, worth noting that this is just the beginning. The industry is already moving beyond FP16/BF16 toward even more aggressive quantization schemes. INT8 training, 4-bit AdamW, exotic formats like FP8 are pushing boundaries of how much numerical precision we can sacrifice while maintaining training effectiveness. As these techniques converge the line between mixed precision training and model quantization is increasingly blurring.

The fundamental insight remains the same: neural networks are remarkably tolerant of numerical approximation, and this tolerance is a feature, not a bug. The same robustness that makes these systems work with approximate arithmetic makes them robust to noise, adversarial examples, and the messy realities of real-world deployment.

## Putting It All Together: A Mixed Precision Playbook

Here's your practical guide for implementing mixed precision in production:

**1. Start Conservative**: Use PyTorch's AutoCast with default settings. Let the framework make the precision decisions until you understand your model's specific requirements.

**2. Monitor Scale Dynamics**: Watch your gradient scaling factor. If it's constantly backing off, you might have numerical stability issues. If it never grows, you might be leaving performance on the table.

**3. Profile Memory AND Compute**: Don't just measure memory savings – measure actual training throughput. Sometimes the communication overhead of mixed precision can hurt more than the compute benefits help.

**4. Test Your Edge Cases**: Run your model with pathological inputs. Mixed precision can expose numerical instabilities that FP32 training masks.

**5. Consider BF16 for Training Stability**: If you're seeing gradient scaling instability, BF16's extended range might be worth the slightly lower precision.

**6. Keep Critical Operations in FP32**: Loss computation, batch normalization, and layer normalization generally should stay in FP32 for numerical stability.

𒅃 The universal Turing machine could perform any computation with infinite precision on an infinite tape. We've decided that's overkill for intelligence, so we're building machines that think approximately, quickly, and efficiently. Turns out God or the universe has a sense of humor about precision, the more precisely we try to compute intelligence, the less intelligent the results seem to become.

Next up: **Gradient Checkpointing: Memory-Compute Tradeoffs** – where we learn to forget activations on purpose and recompute them when needed, because sometimes the best way to remember everything is to remember nothing at all.

---

*Questions about mixed precision? War stories about gradient scaling gone wrong? Subscribers can drop them in the comments. Nothing builds engineering camaraderie like shared suffering over numerical precision edge cases. And if you're one of those people who still insists on FP64 for everything, we need to talk. And not just about how you’re affording your electric bill.*

*The universal machine could simulate any procedure with infinite precision. We've discovered that finite precision is not only sufficient for intelligence, it might actually be preferable. Progress, or a really elaborate form of engineering laziness? The GPUs seem pretty happy about it either way.*