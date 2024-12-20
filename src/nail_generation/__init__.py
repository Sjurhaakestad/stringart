import numpy as np

def generate_nails(detail_map: np.ndarray, base_count=1000, detail_multiplier=2.0) -> list:
    h, w = detail_map.shape

    nails = []
    # Place nails along the border
    # top row
    for x in range(w):
        nails.append((x, 0))
    # bottom row
    for x in range(w):
        nails.append((x, h - 1))
    # left column
    for y in range(1, h - 1):
        nails.append((0, y))
    # right column
    for y in range(1, h - 1):
        nails.append((w - 1, y))

    # Flatten detail map and use it as probabilities for placing inner nails
    flattened = detail_map.flatten()
    # Avoid division by zero if sum is zero
    total = flattened.sum()
    if total == 0:
        # If no detail, just place nails randomly
        normalized_probs = np.ones_like(flattened) / len(flattened)
    else:
        normalized_probs = flattened / total

    # Weighted random selection
    num_inner_nails = base_count
    indices = np.random.choice(len(flattened), size=num_inner_nails, p=normalized_probs)
    for idx in indices:
        y = idx // w
        x = idx % w
        nails.append((x, y))

    return nails
