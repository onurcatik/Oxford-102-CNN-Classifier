# Oxford-102-Dataset & Flower Classification with PyTorch

This repository equips you with the necessary tools to classify images from the Oxford-102 dataset, which contains 102 different categories of flowers.

- [Google Drive Folder](https://drive.google.com/drive/folders/1IskFFmRgiOnzfaGEVHCWF7lz-tTZAOha?usp=sharing)
- [Download the dataset from the provided link](https://drive.google.com/drive/folders/1ylUod4ycahtCdsr1Xc0halkBYaUXuM0-)

## Installation

To get started, ensure you have Python and pip installed on your system. Then, follow these simple steps to install all the necessary dependencies:

1. Clone this repository to your local machine:

```sh
git clone https://github.com/onurcatik/Oxford-102-CNN-Classifier.git
```

2. Navigate to the cloned directory:

```sh
cd Oxford-102-CNN-Classifier
```

3. Install the required dependencies using pip:

```sh
pip install -r requirements.txt
```

## Project Structure

- [Overview](./01_Overview/README.md) - provides a high-level introduction to the project and its goals.
- [CNN Model Class](./02_CNN-Model-Class/README.md) - contains the implementation of the Convolutional Neural Network (CNN) model for image classification.
- [Model Utilities](./03_Model-Utilities/README.md) - includes utility functions and classes used in model training and evaluation.
- [Dataset Download](./04_Dataset-Download/README.md) - guides you through downloading the Oxford-102 dataset for training and testing.
- [Dataloader](./05_Dataloader/README.md) - demonstrates how to load and preprocess data using PyTorch's DataLoader.
- [Traning Loop](./06_Training-Loop/README.md) - explains the process of training the model using backpropagation and optimization techniques.
- [Test Loop](./07_Test-Loop/README.md) - details the evaluation of the trained model on test data.
- [All Together](./09_All-Together/README.md) - houses complete tutorial from [CNN Classifier - Flower Images](https://www.youtube.com/playlist?list=PLhFg2q8pZdMRG4DcqBcBiaDCSVcLNMBpa)
- [Project](./10_Project/training_cnn.ipynb) - houses [Google Colab](https://drive.google.com/drive/folders/1IskFFmRgiOnzfaGEVHCWF7lz-tTZAOha?usp=sharing) project (splited dataset from [original dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), labels, test images from web)

## Source

- [CNN Classifier - Flower Images](https://www.youtube.com/playlist?list=PLhFg2q8pZdMRG4DcqBcBiaDCSVcLNMBpa)
- [PyTorch - Tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
