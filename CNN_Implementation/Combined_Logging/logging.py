# THESE ARE THE LIBRARIES NEEEDED TO RUN THE CNN MODEL IN PYTORCH.
# SKLEARN IS USED TO IMPORT THE DATASET.
# MATPLOTLIB IS USED TO PLOT THE IMAGES.


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import wandb
import config
from torch.utils.tensorboard import SummaryWriter
from clearml import Task, Logger


# WANDB IS INITILIZED HERE
# WE NAME THE PROJECT AS 'mnist-classification'\
# WE DEFINE THE CONFIGURATION PARAMETERS
# WE DEFINE THE LEARNING RATE, EPOCHS, BATCH SIZE, AND ACTIVATION FUNCTION
# WE WILL USE THESE PARAMETERS TO TUNE THE MODEL

wandb.init(project="mnist-classification",
           config = config.config     # THE CONGIGURATION PARAMETERS ARE SAVED IN SEPERATE FILE
)

# THIS IS TENOSRBOARD INITIALIZATION
writer = SummaryWriter('runs/mnist_classification')
task = Task.init(project_name="mnist-classification", task_name="CNN Training")



# THE MNIST DATASET IS IMPORTED FROM THE SKLEARN LIBRARY.
# THE DATASET IS SPLIT INTO TRAINING AND TESTING DATA.

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to tensor
])

# LOADING DATA USING TORCHVISION ONLY
full_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


x_train = full_train_data.data.float() / 255.0
y_train = full_train_data.targets
x_test = full_test_data.data.float() / 255.0
y_test = full_test_data.targets

# Print shapes for verification
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# THE IMAGES ARE NORMALIZED TO A VALUE BETWEEN 0 AND 1.
# THIS IS DONE TO IMPROVE THE TRAINING PROCESS.
# WHAT HAPPENS IF NOT NORMALIZED?

# ----------------------------------------------------------------
# If we do not normalize the data: The learning process may become slower and less efficient.
# as features with larger ranges can dominate gradient updates, leading to slower convergence. 
# This can result in poor model performance, 
# as models using gradient descent may be disproportionately influenced by features with larger scales.
#  Additionally, the lack of normalization complicates hyperparameter. Making it difficult to compare the importance of different features. 
# ----------------------------------------------------------------

# x_train, x_test = x_train / 255.0, x_test / 255.0


# CONVERSION OF NUMPY ARRAYS TO TENSORS IN PYTORCH
# THIS IS NEEDED BECAUSE PYTORCH WORKS WITH TENSORS. 
# IT IS OPTIMIZED FOR CALCULATIONS ON TENSORS.
# IT ALLOWS FOR PARALLEL COMPUTATIONS
# THE TENSORS ARE USED TO CREATE THE DATASET AND DATALOADER IN THE NEXT CODE


x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# CREATING THE DATALOADERS
full_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=1000, shuffle=False)





# THE CNN MODEL IS DEFINED BELOW
# THE CNN MODEL CONSISTS OF THREE CONVOLUTIONAL LAYERS AND TWO FULLY CONNECTED LAYERS
# A MAX POOLING LAYER IS USED TO REDUCE THE DIMENSIONALITY OF THE DATA
# THIS MAX POOL LAYER IS APPLIED BETWEEN THE CONVOLUTIONAL LAYERS AND THE FULLY CONNECTED LAYERS
# A DROPOUT LAYER IS USED TO PREVENT OVERFITTING

# THE FIRST CONV LAYER HAS 32 FILTERS, THE SECOND CONV LAYER HAS 64 FILTERS AND THE THIRD CONV LAYER HAS 128 FILTERS
# THE STRIDE OF 1 MEANS THAT THE FILTER MOVES ONE PIXEL AT A TIME
# THE PADDING OF 1 MEANS THAT THE INPUT IMAGE IS PADDDED WITH ZEROS TO MAINTAIN THE SAME DIMENSIONALITY
# THE KERNEL SIZE OF 3 MEANS THAT THE FILTER SIZE IS 3X3

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
    

    # THIS FUNCTION DESCRIBES THE FORWARD PASS OF THE CNN MODEL
    # THE FORWARD FUNCTION DESCRIBES HOW THE DATA FLOWS THROUGH THE NETWORK
    # THE RELU ACTIVATION FUNCTION IS USED AFTER EACH CONVOLUTIONAL LAYER AND FULLY CONNECTED LAYER
    # THE RELU FUNCTION IS USED TO INTRODUCE NON-LINEARITY INTO THE MODEL

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




# HERE IS THE CODE THAT ITERATES THROUGH THE MODEL AND TRAINS IT
# THE MODEL IS INTILIAZED WITH THE CNN CLASS
# THEN THE FOLLOWING ARE DEFINED:
# THE LOSS FUNCTION (CROSS ENTROPY LOSS)
# THE OPTIMIZER (ADAM OPTIMIZER)
# THE LEARNING RATE (0.001)
# THE NUMBER OF EPOCHS (7)


# THEN EVALUATION IS PERFORMED: 
# THE TRAINING LOOP ITERATES THROUGH THE TRAINING DATA AND UPDATES THE WEIGHTS OF THE MODEL
# THE MODEL IS THEN EVALUATED ON THE VALIDATION DATA TO CHECK FOR OVERFITTING


for epoch in range(config.config["num_epochs"]):
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
    print(f"Epoch [{epoch + 1}/{config.config["num_epochs"]}], Loss: {avg_train_loss:.4f}")

    # LOSSES LOGGED ON EACH 
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    wandb.log({"train_loss": avg_train_loss})
    Logger.current_logger().report_scalar("train_loss", "Loss", iteration=epoch, value=avg_train_loss)

    # VALIDATION STEP
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

    # HERE WE LOG THE VALIDATION LOSS ON EACH
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
    wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy})
    Logger.current_logger().report_scalar("val_loss", "Loss", iteration=epoch, value=avg_val_loss)
    Logger.current_logger().report_scalar("val_accuracy", "Accuracy", iteration=epoch, value=val_accuracy)



def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    images_with_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item(    )
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # THIS CODE LOGS THE IMAGES PROVIDED TO THE MODLE 
            # AND THE PREDICTIONS MADE BY THE MODEL
            # THESE IMAGES ARE DISPLAYED ON THE WANDB DASHBOARD

            for img, label, pred in zip(images.cpu(), labels.cpu(), predicted.cpu()):
                img_np = img.squeeze().numpy()
                images_with_preds.append(
                    wandb.Image(img_np, caption=f"Label: {label}, Pred: {pred}")
                )


    # FINAL METRICS ARE CALCULATED HERE
    # THE ACCURACY, PRECISION, RECALL AND F1 SCORE ARE CALCULATED
    # THESE METRICS ARE USED TO EVALUATE THE PERFORMANCE
    # OF THE MODEL ON THE TEST DATA
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # LHERE WE DIFINE WHAT METRICS TO LOG TO WANDB
    # THESE METRICS ARE DISPLAYED ON THE WANDB DASHBOARD
    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "images_with_predictions": images_with_preds
    })

    # TENSORBOARD METRICS ARE LOGGED HERE
    writer.add_scalar('Accuracy/test', accuracy)
    writer.add_scalar('Precision/test', precision)
    writer.add_scalar('Recall/test', recall)
    writer.add_scalar('F1_Score/test', f1)

    
    # METRICS LOGGED TO CLEARML
    Logger.current_logger().report_scalar("test_accuracy", "Accuracy", iteration=0, value=accuracy)
    Logger.current_logger().report_scalar("test_precision", "Precision", iteration=0, value=precision)
    Logger.current_logger().report_scalar("test_recall", "Recall", iteration=0, value=recall)
    Logger.current_logger().report_scalar("test_f1_score", "F1 Score", iteration=0, value=f1)



    # THE CONFUSION MATRIX SHOWS HOW MANY IMAGES WERE CORRECTLY CLASSIFIED
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    cm_display.plot()
    plt.savefig("confusion_matrix.png")

    # CONFUSION MATRIX LOGGED HERE
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    writer.add_figure('Confusion Matrix', cm_display.figure_)
    Logger.current_logger().report_image("Confusion Matrix", "Confusion Matrix", iteration=0, local_path="confusion_matrix.png")
    

evaluate(model, test_loader)

# CLOSING
wandb.finish()
writer.close()
task.close()





