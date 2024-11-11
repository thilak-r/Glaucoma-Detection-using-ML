from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

app = Flask(__name__)

# Define the GlaucomaNet model
class GlaucomaNet(nn.Module):
    def __init__(self):
        super(GlaucomaNet, self).__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        num_classes = 2  # Number of classes (glaucoma and normal)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout layer

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return x

# Load the trained model
model = GlaucomaNet()

model.load_state_dict(torch.load('Glaucoma_model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()  # Set the model to evaluation mode

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file format"})

        # Save the uploaded image to the uploads folder
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        with torch.no_grad():
            # Get model output
            output = model(image)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            
            # Get the predicted class and its score
            _, predicted = torch.max(output, 1)
            score = probabilities[0, predicted].item()
            
            # Define class names
            classes = ['glaucoma', 'normal']
            prediction = classes[predicted.item()]
            
            return jsonify({"prediction": prediction, "score": score})
    
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
