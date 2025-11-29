from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

MOCK_DETECTIONS = [
    {"id": 1, "type": "excavator", "confidence": 0.92, "location": "51.1079,17.0385"},
    {"id": 2, "type": "crane", "confidence": 0.88, "location": "51.1085,17.0390"}
]

MOCK_PROGRESS = {
    "percent_complete": 42,
    "notes": "Earthworks phase ongoing. Foundation preparation in progress."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "no files field"}), 400
    saved = []
    for f in request.files.getlist('files'):
        if f.filename == '':
            continue
        filename = secure_filename(f.filename)
        # make unique
        base, ext = os.path.splitext(filename)
        ts = int(time.time()*1000)
        unique = f"{base}_{ts}{ext}".lower()
        path = os.path.join(UPLOAD_DIR, unique)
        f.save(path)
        saved.append(unique)
    return jsonify({"status": "ok", "saved": saved}), 200

@app.route('/images')
def images():
    files = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp')):
            files.append(name)
    return jsonify(files)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    # sanitize
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    try:
        os.remove(path)
    except OSError:
        return jsonify({"error": "delete failed"}), 500
    return jsonify({"status": "deleted", "filename": safe})

@app.route('/detections')
def detections():
    return jsonify(MOCK_DETECTIONS)

@app.route('/progress')
def progress():
    return jsonify(MOCK_PROGRESS)

@app.route('/report')
def report():
    report = {
        "detections": MOCK_DETECTIONS,
        "progress": MOCK_PROGRESS,
        "summary": "Automated report generated successfully."
    }
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, port=5000)