import os
import nltk
from flask import Flask, request, jsonify

# Add this to ensure NLTK downloads required data
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))  # Set custom data path
nltk.download('punkt', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    # Your classification logic here
    try:
        data = request.json
        text = data.get('text')
        # Add your classification logic using nltk here
        return jsonify({"result": "Classification successful"})
    except Exception as e:
        return jsonify({"error": f"Error during processing: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
