import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import skimage.io
import matplotlib.pyplot as plt

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

# Define image transformations
flower_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Create dataset and DataLoaders
dataset = MyFlowerDataset(metadata, transform=flower_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Visualize transformed images
for images, labels in train_loader:
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    for i in range(2):
        for j in range(4):
            img = images[i * 4 + j].cpu().permute(1, 2, 0)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    break  # Only show one batch of images
plt.show()
