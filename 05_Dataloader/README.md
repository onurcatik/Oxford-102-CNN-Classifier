# Writing PyTorch Dataset and DataLoader

In the previous part, we downloaded the data, visualized it, and observed variations in image sizes and shapes, which are crucial to address as neural networks require consistent image dimensions. We defined metadata for the dataset, and now we'll focus on writing a Dataset class and defining DataLoaders.

## Writing PyTorch Dataset Class

In PyTorch, when creating a custom dataset, you need to define three crucial methods:

1. `__init__`: Initializes the dataset.
2. `__len__`: Returns the size of the dataset.
3. `__getitem__`: Retrieves a sample from the dataset.

Let's start by defining the `MyFlowerDataset` class:

```python
import torch
from torch.utils.data import Dataset

class MyFlowerDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        image_path = self.metadata.iloc[index, 0]
        image = skimage.io.imread(image_path)
        label = self.metadata.iloc[index, 1]
        label = torch.tensor(label)
  
        if self.transform:
            image = self.transform(image)
  
        return image, label
```

## Image Transformation

Before proceeding, let's define the transformations we want to apply to our images. We'll use `torchvision.transforms` for this purpose.

```python
from torchvision import transforms

flower_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```

## Creating Dataset and DataLoaders

Now, let's create our dataset and split it into training and testing sets using `torch.utils.data.random_split`. Then, we'll define DataLoaders for both sets.

```python
from torch.utils.data import DataLoader, random_split

dataset = MyFlowerDataset(metadata, transform=flower_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

## Visualizing Transformed Images

Now, let's visualize the transformed images from the dataset using Matplotlib:

```python
import matplotlib.pyplot as plt

# Fetching a batch of images from the train loader
for images, labels in train_loader:
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    for i in range(2):
        for j in range(4):
            img = images[i * 4 + j].cpu().permute(1, 2, 0)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    break  # Only show one batch of images
plt.show()
```
