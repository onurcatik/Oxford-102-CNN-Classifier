import torch
import torch.optim as optim
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create an instance of the CNN model
num_classes = 10  # Example: If you're working on CIFAR-10, set it to 10
model = SimpleCNN(num_classes)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print the model architecture
print(model)

# Define hyperparameters
NUM_OUT_CHANNELS = (8, 16)  # Number of output channels for convolutional layers
IMAGE_WIDTH = 224           # Input image width
IMAGE_HEIGHT = 224          # Input image height
BATCH_SIZE = 32             # Mini-batch size
NUM_EPOCHS = 4              # Number of training epochs
LEARNING_RATE = 0.001       # Learning rate for optimizer

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define loss function
criterion = nn.CrossEntropyLoss()
