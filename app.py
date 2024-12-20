import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from src.image_processing import load_image, to_grayscale, detect_edges, compute_detail_intensity
from src.nail_generation import generate_nails
import cv2
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get parameters from form
        low_thresh = int(request.form.get('low_thresh', 100))
        high_thresh = int(request.form.get('high_thresh', 200))
        base_count = int(request.form.get('base_count', 200))
        detail_multiplier = float(request.form.get('detail_multiplier', 2.0))
        
        # Save uploaded file
        file = request.files['image_file']
        if file.filename == '':
            return "No file selected", 400
        image_path = os.path.join('data', file.filename)
        file.save(image_path)

        # Process image
        img = load_image(image_path)
        gray = to_grayscale(img)
        edges = detect_edges(gray, low_thresh, high_thresh)
        cv2.imwrite(os.path.join('data', 'edges.jpeg'), edges)

        detail_map = compute_detail_intensity(edges)
        # Visualize detail map by scaling to 0-255
        detail_map_vis = (detail_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('data', 'detail_map.jpeg'), detail_map_vis)

        nails = generate_nails(detail_map, base_count, detail_multiplier)
        # Draw nails on original image
        vis = img.copy()
        h, w, _ = vis.shape
        for (x, y) in nails:
            color = (0, 0, 255) if (y == 0 or y == h-1 or x == 0 or x == w-1) else (255, 0, 0)
            cv2.circle(vis, (x, y), 3, color, -1)
        cv2.imwrite(os.path.join('data', 'nails_visualization.jpeg'), vis)

        return redirect(url_for('results', filename='nails_visualization.jpeg'))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    app.run(debug=True)
