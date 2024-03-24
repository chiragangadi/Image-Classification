
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torchvision.models as models

num_classes = 8
batch_size = 256
num_epochs = 75
learning_rate = 0.001

# Define the path to the root folder containing subfolders for each class
root_folder = "/Users/chiragangadi/Uni Siegen/03_DL/Exersice/task/data/train/"

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),   # Resize the image to a fixed size
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
])

# Create an ImageFolder dataset
dataset = datasets.ImageFolder(root=root_folder, transform=transform)

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create a DataLoader to handle batching and shuffling of the data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet-101 model
Net = models.resnet101(pretrained = True)

for param in Net.parameters():
    param.requires_grad = False

# Modify final fully connected layer to have 8 outputs
num_ftrs = Net.fc.in_features
Net.fc = nn.Linear(num_ftrs, num_classes)  # Modify the last layer 
Net.fc.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Net.fc.parameters(), lr=learning_rate)
train_loss = []
val_loss = []

# Training loop
for epoch in range(num_epochs):
    Net.train()
    for batch_images, batch_labels in train_loader:
        batch_labels_onehot = F.one_hot(batch_labels, num_classes=num_classes)
        
        # Forward pass
        outputs = Net(batch_images)

        # Compute the loss
        lossT = criterion(outputs, batch_labels_onehot.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        lossT.backward()
        optimizer.step()
    
    train_loss.append(lossT.item())
    
    Net.eval()
    with torch.no_grad():
        for batch_images, batch_labels in val_loader: 
            batch_labels_onehot = F.one_hot(batch_labels, num_classes=num_classes)
            outputs = Net(batch_images)
            lossV = criterion(outputs, batch_labels_onehot.float())
    
        val_loss.append(lossV.item())
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {lossT.item():.4f}, Validation Loss: {lossV.item():.4f}")

print("Training finished.")



# Plot loss vs epoch
plt.plot(range(1, num_epochs+1), train_loss, label='Training Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(1, num_epochs+1), val_loss, label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
