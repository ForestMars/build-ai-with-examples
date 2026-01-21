# binding_robustness.py - Computes Binding Robustness Score

def compute_binding_robustness(model, tokenizer, test_cases, 
                               distractor_lengths=[0, 512, 2048, 8192]):
    """
    Test binding stability as distractors are injected.
    
    test_cases: List of (entity, property, correct_value) tuples
    distractor_lengths: Context pollution levels
    """
    results = []
    
    for entity, property_name, correct_value in test_cases:
        robustness_scores = []
        
        for dist_len in distractor_lengths:
            # Generate semantic distractors (related but irrelevant context)
            distractor_text = generate_semantic_distractors(
                entity, property_name, target_length=dist_len
            )
            
            # Build test prompt
            prompt = f"{distractor_text}\n\nQuestion: What is the {property_name} of {entity}?"
            
            # Get model prediction
            output = model.generate(tokenizer.encode(prompt, return_tensors="pt"))
            prediction = extract_answer(tokenizer.decode(output[0]))
            
            # Score correctness
            is_correct = (prediction == correct_value)
            robustness_scores.append({
                'distractor_length': dist_len,
                'correct': is_correct,
                'prediction': prediction
            })
        
        # Compute robustness: how long until binding breaks?
        breaking_point = next(
            (s['distractor_length'] for s in robustness_scores if not s['correct']),
            max(distractor_lengths)
        )
        
        results.append({
            'entity': entity,
            'property': property_name,
            'breaking_point': breaking_point,
            'scores': robustness_scores
        })
    
    return results
