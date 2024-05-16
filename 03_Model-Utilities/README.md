# Writing Model Utilities in PyTorch

Welcome to the second video in our series on building a convolutional neural network (CNN) for image classification using PyTorch. In the previous video, we started by writing the CNN model class and tested it. In this video, we will focus on defining hyperparameters, device settings, optimizer, and loss function. Let's dive in!

## Step 1: Define Hyperparameters

Hyperparameters are crucial settings that determine the architecture and behavior of our model. These include:

- Number of output channels
- Image width and height
- Batch size
- Number of epochs
- Learning rate

```python
# Define hyperparameters
NUM_OUT_CHANNELS = (8, 16)  # Number of output channels for convolutional layers
IMAGE_WIDTH = 224           # Input image width
IMAGE_HEIGHT = 224          # Input image height
BATCH_SIZE = 32             # Mini-batch size
NUM_EPOCHS = 4              # Number of training epochs
LEARNING_RATE = 0.001       # Learning rate for optimizer
```

## Step 2: Define Device

We need to specify whether to use CUDA (GPU) or CPU for computation. This is essential for leveraging GPU acceleration if available.

```python
import torch

# Define device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## Step 3: Define Optimizer

We'll use the Adam optimizer, which is widely used for training neural networks. It requires model parameters and a learning rate.

```python
import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

## Step 4: Define Loss Function

For classification tasks, cross-entropy loss is commonly used. It measures the difference between predicted probabilities and actual class labels.

```python
import torch.nn as nn

# Define loss function
criterion = nn.CrossEntropyLoss()
```
