from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl file not found. Ensure the model file is in the correct directory.")
    model = None

# Define the class labels (update these based on your dataset)
class_labels = ['peper', 'e-waste', 'Organic', 'Plastic', 'Metal', 'Glass']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the uploaded image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize to match the model's input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Check if the model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check the model.pkl file.'}), 500

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return jsonify({'result': f'Your waste is classified as {predicted_class}.'})
    except Exception as e:
        return jsonify({'error': f'An error occurred during classification: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)