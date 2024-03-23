
test_root = "/Users/chiragangadi/Uni Siegen/03_DL/Exersice/task/data/val/"

# Define transformations for the testing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create an ImageFolder dataset for test
test_dataset = ImageFolder(root=test_root, transform=transform)

# Create a DataLoader for validation data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

Net.eval()  # Set the model to evaluation mode

# Define a list to store predictions and true labels
predictions = []
true_labels = []

# Perform inference on the testing set
with torch.no_grad():
    for image, labels in test_loader:

        # Forward pass
        output = Net(image)
        
        # Get the predicted label
        predicted_label = torch.argmax(output)
        predictions.append(predicted_label.item())
        
        label = F.one_hot(labels, num_classes=num_classes)
        label = torch.argmax(label)
        true_labels.append(label.item())

# Evaluate the accuracy
correct_predictions = 0;
for i in range(len(predictions)):
    if predictions[i]==true_labels[i]:
        correct_predictions +=1
        
total_samples = len(test_loader.dataset)
accuracy = correct_predictions / total_samples

print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
