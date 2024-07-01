import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, send_from_directory

# THIS IS THE SAME CODE AS WE HAD EMPLOYED BEFORE.
# WE HAD SAVED THE MODEL PARAMETERS IN THE 'model' DIRECTORY
# WE WILL LOAD THEM FROM THERE
# THE WEIGHTS AND BIAS WILL BE USED 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        x = self.pool(torch.tanh(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# HERE WE UPLOAD THE MODEL PARAMETERS
model = torch.load('model/mnist_cnn.pth', map_location=torch.device('cpu'))

# THIS IS HOW WE WILL TRANSFORM THE INPUT IMAGE
# USER CAN INPUT ANY SHAPE AND SIZE OF IMAGE
# WE WILL CONVERT IT TO 28x28 GRAYSCALE IMAGE
# AND THEN NORMALIZE IT

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


app = Flask(__name__)

# UPLOAD_FOLDER WILL STORE THE UPLOADED IMAGES
UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# THIS FUNCTION WILL RENDER THE IMAGE 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file provided')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = Image.open(file_path)
        img = transform(img)
        img_tensor = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = predicted.item()

        return render_template('result.html', predicted_class=predicted_class, image_file=file.filename)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)