import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load fertilizer suggestions (tab-separated)
fertilizer_map = {}
with open('fertilizer_suggestions.txt', 'r', encoding='utf-8') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')  # ensure it's tab-separated
        if len(parts) == 3:
            disease, suggestion, quantity = parts
            fertilizer_map[disease.strip()] = (suggestion.strip(), quantity.strip())


with open('label_map.txt', 'r') as f:
    labels = f.read().splitlines()

num_classes = len(labels)  # ✅ Automatically adjust to label count


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)


# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()

            # ✅ Safety check
            if class_idx >= len(labels):
                return jsonify({'error': 'Prediction index out of range'}), 500

            class_name = labels[class_idx]

        suggestion, quantity = fertilizer_map.get(class_name, ('No suggestion available', 'N/A'))

        return jsonify({
            'image_path': '/' + filepath,
            'disease': class_name,
            'suggestion': suggestion,
            'quantity': quantity
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
