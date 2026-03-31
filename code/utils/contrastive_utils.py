import random


# Template pools for LLM-assisted semantic expansion of hydrological descriptions
TURBIDITY_TEMPLATES = [
    "a satellite image of a river with {level} turbidity",
    "remote sensing imagery showing {level} suspended sediment concentration in a fluvial system",
    "a multispectral snapshot of a {morphology} river with {level} sediment load",
    "aerial view of a {region} river channel exhibiting {level} water turbidity",
    "satellite observation of {level} sediment-laden water in a {morphology} river network",
    "optical remote sensing image capturing {level} suspended sediments in surface water",
    "a river scene with {level} turbid plumes visible from satellite imagery",
    "multispectral reflectance image of a {morphology} river with {level} SSC",
]

MORPHOLOGY_POOL = ["braided", "meandering", "confined", "anastomosing", "straight"]
REGION_POOL = ["Tibetan Plateau", "high-altitude", "mountainous", "alpine"]


def generate_text_descriptions(ssc_values, stations=None, num_variants=4):
    """
    Generate diverse hydrological text descriptions for each sample using
    LLM-assisted semantic expansion.

    Args:
        ssc_values (list or np.array): Ground-truth SSC values in g/m^3.
        stations (list, optional): Station names or identifiers.
        num_variants (int): Number of text variants to generate per sample.

    Returns:
        list: A list of text descriptions (one per sample).
    """
    descriptions = []
    for i, ssc in enumerate(ssc_values):
        # Determine concentration level
        if ssc < 50:
            level = "low"
        elif ssc < 200:
            level = "moderate"
        elif ssc < 500:
            level = "high"
        else:
            level = "extreme"
        
        variants = []
        for _ in range(num_variants):
            template = random.choice(TURBIDITY_TEMPLATES)
            morphology = random.choice(MORPHOLOGY_POOL)
            region = random.choice(REGION_POOL)
            desc = template.format(level=level, morphology=morphology, region=region)
            variants.append(desc)
        
        # Select one variant randomly (or concatenate if desired)
        descriptions.append(random.choice(variants))
    
    return descriptions


def split_head_tail(labels, head_ratio=0.75):
    """
    Split dataset indices into head and tail distributions based on SSC values.

    Args:
        labels (list or np.array): SSC labels.
        head_ratio (float): Proportion of samples assigned to the head distribution.

    Returns:
        tuple: (head_indices, tail_indices) as numpy arrays.
    """
    import numpy as np
    labels = np.array(labels)
    threshold = np.percentile(labels, head_ratio * 100)
    head_indices = np.where(labels <= threshold)[0]
    tail_indices = np.where(labels > threshold)[0]
    return head_indices, tail_indices
