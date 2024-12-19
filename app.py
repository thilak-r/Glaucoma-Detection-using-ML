from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "best_model.pth"

# Initialize the model architecture
model = models.resnet18(pretrained=False)  # Use the same architecture as during training
model.fc = nn.Linear(model.fc.in_features, 3)  # Ensure the final layer matches the number of classes

# Load the state_dict
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Class names corresponding to the labels
CLASSES = ['Glaucoma', 'Normal', 'Unknown']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']

    try:
        # Open the image
        image = Image.open(file).convert('RGB')

        # Preprocess the image
        image = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            confidence, predicted_class = torch.max(probabilities, 0)

        # Return prediction and confidence score
        return jsonify({
            'prediction': CLASSES[predicted_class.item()],
            'score': confidence.item()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensuring the app runs on the correct host and port provided by Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
