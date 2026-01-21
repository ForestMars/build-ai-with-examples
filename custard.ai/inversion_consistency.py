inversion_consitency.py - Check if the model flips the same co-hyponym pairs across domains.

cohyponym_pairs = {
    'biology': [
        ('mitochondria', 'chloroplast', 'ATP_synthesis'),
        ('dendrite', 'axon', 'receives_signals')
    ],
    'chemistry': [
        ('anode', 'cathode', 'oxidation'),
        ('acid', 'base', 'proton_donor')
    ],
    'physics': [
        ('conductor', 'insulator', 'free_electrons'),
        ('convex', 'concave', 'diverging')
    ]
}

# If inversion rate is consistent across domains, it's architectural
# If domain-specific, it's a data gap
