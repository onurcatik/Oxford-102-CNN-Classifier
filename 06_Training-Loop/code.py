import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define your CNN model class (not shown in the provided context)
# Assuming model class is already defined

# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 32

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming you have created train_loader and test_loader
# Load Dataset and Create Data Loaders
# Assuming train_loader and test_loader are defined

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for index, (inputs, labels) in enumerate(train_loader):
        # Send data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
  
        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
