from flask import Flask, request, jsonify
from model import compute_similarity
import os

app = Flask(__name__)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json(force=True, silent=True)
    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({"error": "Request body must contain 'text1' and 'text2'"}), 400

    text1 = data['text1']
    text2 = data['text2']
    score = compute_similarity(text1, text2)
    return jsonify({"similarity score": score})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Semantic Similarity API is running!"})

if __name__ == '__main__':
    # For local debugging only. Render/Gunicorn will set PORT and run the app.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
