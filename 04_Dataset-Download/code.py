import os
import random
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io as skio
# from google.colab import drive

# Mount Google Drive
# drive.mount('/content/drive')

# Define data and label paths
data_path = '/content/drive/My Drive/102 flowers/'
label_path = os.path.join(data_path, 'image_labels.mat')

# Load labels
labels = sio.loadmat(label_path)['labels'] - 1  # Convert to Python indexing

# Generate random image numbers
image_num = random.sample(range(1, 8190), 8)

# Create subplots for visualization
fig, axes = plt.subplots(2, 4, figsize=(8, 12))

for i in range(2):
    for j in range(4):
        # Generate image path
        image_path = os.path.join(data_path, f"image_{image_num[i*4 + j]:05d}.jpg")
  
        # Read and display image
        image = skio.imread(image_path)
        axes[i, j].imshow(image)
        axes[i, j].axis('off')
        axes[i, j].set_title(f"Label: {labels[image_num[i*4 + j]]}")

plt.tight_layout()
plt.show()

# Generate image paths
image_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

# Create metadata DataFrame
metadata = pd.DataFrame({'image_path': image_paths, 'label': labels.flatten()})
