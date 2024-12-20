import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {path} not found.")
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_edges(gray_image: np.ndarray, low_thresh=100, high_thresh=200) -> np.ndarray:
    edges = cv2.Canny(gray_image, low_thresh, high_thresh)
    return edges

def compute_detail_intensity(edges: np.ndarray) -> np.ndarray:
    # Convert edges (0 or 255) to float in range [0,1]
    detail_map = edges.astype(np.float32) / 255.0
    # Optional: blur to create a smoother intensity distribution
    detail_map = cv2.GaussianBlur(detail_map, (5, 5), 0)
    return detail_map
