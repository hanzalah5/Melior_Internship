# THIS IS CNN MODEL TRAINING USING CLEARML

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from clearml import Task, Logger

# CLEARML IS INTILIZED HERE WITH THE TASK NAME AND PROJECT NAME
task = Task.init(project_name="mnist-classification", task_name="CNN Training")

# THIS IS THE SAME CODE AS BEFORE

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert numpy arrays to tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets and data loaders
full_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=1000, shuffle=False)

# Define CNN model
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

# THIS IS THE CHECK FOR GPU AVALABILITY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TRAIN THE CNN MODEL
num_epochs = 7
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
    Logger.current_logger().report_scalar("train_loss", "Loss", iteration=epoch, value=avg_train_loss)

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    Logger.current_logger().report_scalar("val_loss", "Loss", iteration=epoch, value=avg_val_loss)
    Logger.current_logger().report_scalar("val_accuracy", "Accuracy", iteration=epoch, value=val_accuracy)

# Evaluate the model
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # HERE WE LOG THE METRIC TO CLEARML 
    Logger.current_logger().report_scalar("test_accuracy", "Accuracy", iteration=0, value=accuracy)
    Logger.current_logger().report_scalar("test_precision", "Precision", iteration=0, value=precision)
    Logger.current_logger().report_scalar("test_recall", "Recall", iteration=0, value=recall)
    Logger.current_logger().report_scalar("test_f1_score", "F1 Score", iteration=0, value=f1)

    # HERE WE LOG THE CONFUSION MATRIX TO CLEARML
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    cm_display.plot()
    plt.savefig("confusion_matrix.png")
    Logger.current_logger().report_image("Confusion Matrix", "Confusion Matrix", iteration=0, local_path="confusion_matrix.png")


evaluate(model, test_loader)

# CLEARML TASK IS CLOSED HERE
task.close()
