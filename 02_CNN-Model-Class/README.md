# Writing a CNN Model Class in PyTorch

In this tutorial, we'll explore crafting a Convolutional Neural Network (CNN) model for image classification from scratch using PyTorch. But before we start writing the model class, let's import the necessary packages.


## 1. Importing Packages

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
```

## 2. Writing the model class

Now that we have imported all the required packages, let's define our model class. We'll name it `MyCNN` and inherit from `nn.Module`.

```python
class MyCNN(nn.Module):
    def __init__(self, num_channels=3, num_out_channels=[8, 16], image_width=100, image_height=100, num_classes=102):
        super(MyCNN, self).__init__()
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes

        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_out_channels[0], kernel_size=3, stride=1, padding=1)
    
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.num_out_channels[0], out_channels=self.num_out_channels[1], kernel_size=3, stride=1, padding=1)
    
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # Calculate input features for fully connected layer
        conv_output_width = self.image_width // 4
        conv_output_height = self.image_height // 4
        self.fc_input_features = conv_output_width * conv_output_height * self.num_out_channels[-1]

        # Define fully connected layer
        self.fc = nn.Linear(self.fc_input_features, self.num_classes)

    def forward(self, x):
        # Convolutional layer 1 with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Convolutional layer 2 with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for fully connected layer
        x = x.view(-1, self.fc_input_features)

        # Fully connected layer
        x = self.fc(x)

        return x
```

Now, let's test our model to ensure it works as expected.

```python
# Instantiate the model
model = MyCNN()

# Generate random input images
x = torch.randn(32, 3, 100, 100)

# Pass input through the model
y = model(x)

# Print output shape
print(y.shape)  # Expected output shape: (32, 102)
```
