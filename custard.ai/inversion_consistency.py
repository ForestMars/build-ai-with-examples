inversion_consitency.py - Check if the model flips the same co-hyponym pairs across domains.

def test_cross_domain_consistency(model, tokenizer, cohyponym_pairs):
    """
    Test if inversion rates are consistent across domains.
    
    If consistent: architectural problem
    If domain-specific: data gap
    """
    results_by_domain = {}
    
    for domain, pairs in cohyponym_pairs.items():
        domain_results = []
        
        for entity_a, entity_b, property_term in pairs:
            # Test both directions
            prompt_a = f"The {property_term} is associated with the {entity_a}"
            prompt_b = f"The {property_term} is associated with the {entity_b}"
            
            # Measure which entity the model prefers for this property
            query = f"Which has {property_term}: {entity_a} or {entity_b}?"
            output = model.generate(
                tokenizer.encode(query, return_tensors="pt"),
                max_new_tokens=50
            )
            response = tokenizer.decode(output[0])
            
            # Check if answer is correct (would need ground truth lookup)
            is_correct = verify_answer(response, entity_a, entity_b, property_term)
            
            domain_results.append({
                'pair': (entity_a, entity_b),
                'property': property_term,
                'correct': is_correct
            })
        
        inversion_rate = 1 - (sum(r['correct'] for r in domain_results) / len(domain_results))
        results_by_domain[domain] = {
            'inversion_rate': inversion_rate,
            'n_pairs': len(domain_results),
            'details': domain_results
        }
    
    # Analyze variance across domains
    rates = [r['inversion_rate'] for r in results_by_domain.values()]
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    
    is_architectural = (std_rate / mean_rate) < 0.3  # Low variance = architectural
    
    return {
        'by_domain': results_by_domain,
        'mean_inversion_rate': mean_rate,
        'std_inversion_rate': std_rate,
        'likely_architectural': is_architectural
    }
