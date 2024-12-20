import os
import cv2
import numpy as np
from src.image_processing import load_image, to_grayscale, detect_edges, compute_detail_intensity
from src.nail_generation import generate_nails

def main():
    image_path = 'data/img1.jpeg'
    if not os.path.exists(image_path):
        raise FileNotFoundError("Please place img1.jpeg in the data/ directory.")

    # Step 1: Image processing
    img = load_image(image_path)
    gray = to_grayscale(img)
    edges = detect_edges(gray, low_thresh=100, high_thresh=200)
    detail_map = compute_detail_intensity(edges)

    # Step 2: Nail generation
    nails = generate_nails(detail_map, base_count=200, detail_multiplier=2.0)
    print(f"Total nails placed: {len(nails)}")
    
    # Count how many are border vs inside
    h, w, _ = img.shape
    border_count = sum((y == 0 or y == h-1 or x == 0 or x == w-1) for x, y in nails)
    inner_count = len(nails) - border_count
    print(f"Nails on border: {border_count}, inner nails: {inner_count}")

    # Check a few inner nails
    inner_nails = [(x,y) for (x,y) in nails if x not in (0,w-1) and y not in (0,h-1)]
    print("Example inner nails (up to 5):", inner_nails[:5])

    # For visualization: draw nails larger
    vis = img.copy()
    for (x, y) in nails:
        # Draw inner nails larger and in blue to differentiate from border nails (red)
        color = (0, 0, 255) if (y == 0 or y == h-1 or x == 0 or x == w-1) else (255, 0, 0)
        cv2.circle(vis, (x, y), 3, color, -1)

    cv2.imwrite('data/nails_visualization.jpeg', vis)
    print("Nail placement visualization saved to data/nails_visualization.jpeg")

if __name__ == '__main__':
    main()
