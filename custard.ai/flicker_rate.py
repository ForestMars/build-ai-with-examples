# flicker_rate.py - Measure output variance across temperature-sampled generations.

def measure_flicker_rate(model, tokenizer, prompt, n_samples=50):
    """
    Generate many samples and measure answer distribution.
    
    High variance indicates unresolved ambiguity in latent space.
    """
    answers = []
    
    for _ in range(n_samples):
        output = model.generate(
            tokenizer.encode(prompt, return_tensors="pt"),
            do_sample=True,
            temperature=0.7  # Not too low (deterministic) or high (random)
        )
        answer = extract_answer(tokenizer.decode(output[0]))
        answers.append(answer)
    
    # Compute distribution
    from collections import Counter
    distribution = Counter(answers)
    
    # Entropy of answer distribution
    total = len(answers)
    probs = [count/total for count in distribution.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    
    # Check for bimodality (two dominant answers with similar frequency)
    top_two = distribution.most_common(2)
    if len(top_two) == 2:
        ratio = top_two[1][1] / top_two[0][1]  # Second most common / most common
        is_bimodal = ratio > 0.4  # Threshold tunable
    else:
        is_bimodal = False
    
    return {
        'distribution': distribution,
        'entropy': entropy,
        'is_bimodal': is_bimodal,
        'flicker_rate': 1 - (distribution.most_common(1)[0][1] / total)
    }
