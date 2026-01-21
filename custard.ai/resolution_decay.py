# resolution_decay.py - Measure resolution decay at various context depths 

def measure_resolution_decay(model, tokenizer, fact, context_lengths):
    """
    Measure how attention entropy increases with context length.
    
    High entropy = model is uncertain which token to attend to
    """
    results = []
    
    for ctx_len in context_lengths:
        # Place fact at position ctx_len
        context = generate_filler_context(ctx_len)
        prompt = f"{context}\n{fact}\nQuestion: [related to fact]"
        
        # Get attention weights
        with torch.no_grad():
            outputs = model(
                tokenizer.encode(prompt, return_tensors="pt"),
                output_attentions=True
            )
        
        # Analyze final layer attention
        final_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        
        # Compute entropy of attention distribution
        attn_dist = final_attn.mean(dim=0)[-1, :]  # Average over heads, final query
        entropy = -torch.sum(
            attn_dist * torch.log(attn_dist + 1e-9)
        ).item()
        
        # Test if model can still answer correctly
        answer_prompt = prompt + "\nAnswer:"
        prediction = model.generate(tokenizer.encode(answer_prompt, return_tensors="pt"))
        is_correct = verify_answer(prediction, fact)
        
        results.append({
            'context_length': ctx_len,
            'attention_entropy': entropy,
            'correct': is_correct,
            'attention_concentration': attn_dist.max().item()
        })
    
    return results
