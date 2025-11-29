from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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