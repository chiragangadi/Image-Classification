
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

# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*28*28, 256)
        self.fc2 = nn.Linear(256,num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 128*28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    

# Set hyperparameters
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

# Create an instance of the model
Net = CNN(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
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
