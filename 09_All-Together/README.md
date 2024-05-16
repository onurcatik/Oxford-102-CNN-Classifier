# CNN Classifier for Image Classification in PyTorch

This repository contains steps towards writing a Convolutional Neural Network (CNN) classifier for image classification in PyTorch. Below are the steps involved in the process:

## Step 1: Writing the Model

In this step, we define the architecture of the CNN model. The `MyCNN` class is implemented which includes convolutional layers, max-pooling layers, and fully connected layers. This class is responsible for the forward pass of the model.

## Step 2: Setting Hyperparameters, Device, Model, Optimizer, and Loss Function

Here, we set the hyperparameters such as the number of output channels, image width and height, batch size, number of epochs, and learning rate. We also define the device for training (GPU if available, otherwise CPU), initialize the model, optimizer (Adam), and loss function (CrossEntropyLoss).

## Step 3: Downloading the Data and Visualizing It

The dataset used for this project consists of 8189 images belonging to 102 different flower species. We provide links to download the dataset and visualize a sample of the images along with their corresponding labels.

## Step 4: Writing Dataset Class, Defining Dataset and DataLoader

In this step, we define a custom dataset class `MyFlowerDataset` for loading the images and their labels. We also define transformations such as resizing, normalization, etc., and split the dataset into training and testing sets. Finally, we create DataLoader objects for efficient data loading during training.

## Step 5: Writing the Training Loop and Training the Model

This step involves writing the training loop where we iterate over the dataset, perform forward and backward passes, update model parameters, and calculate the loss. We train the model for a specified number of epochs and monitor the loss.

## Step 6: Adding Check Accuracy Function and Model Evaluation during Training

Here, we implement a function to calculate the accuracy of the model on the test dataset during training. This function evaluates the model's performance on the test set and prints the accuracy after each epoch.

## Step 7: Adding Progress Bar to the Training Loop

Finally, we enhance the training loop by adding a progress bar to monitor the training progress batch-wise. This provides a visual indication of the training progress and helps in tracking the loss.
