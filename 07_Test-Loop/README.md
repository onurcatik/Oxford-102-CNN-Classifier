# Writing the Test Loop in PyTorch

In this tutorial, we'll focus on writing the test loop to evaluate the performance of our trained model on a test dataset. This is crucial for assessing how well our model generalizes to unseen data. In our previous part, we implemented the training loop for our CNN classifier. While we observed a decrease in the loss value during training, we did not evaluate the accuracy of our model on the test dataset. Here, we'll write a function called `check_accuracy` to handle this task.

### Writing the Test Loop:

First, let's define the `check_accuracy` function. This function will take a data loader (`loader`) and the model (`model`) as inputs. It will calculate the accuracy of the model on the provided dataset.

```python
import torch

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)  # Send data to device (e.g., GPU)
            y = y.to(device)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples
    print(f'Accuracy: {accuracy:.2%}')

    model.train()  # Set model back to training mode
```

Now, let's break down the test loop:

1. **Setting up**: We set the model to evaluation mode using `model.eval()` and disable gradient calculation with `torch.no_grad()` to speed up computation.
2. **Iterating through the dataset**: We loop through the provided data loader, loading batches of data (`x`) and their corresponding labels (`y`).
3. **Forward pass**: We pass the input data through the model to get the output scores. We use `torch.max()` to get the predicted class labels (`predictions`).
4. **Calculating accuracy**: We compare the predicted labels with the ground truth labels (`y`) to count the number of correct predictions (`num_correct`) and the total number of samples (`num_samples`).
5. **Printing accuracy**: Finally, we calculate the accuracy as the ratio of correct predictions to total samples and print it.

### Running the Test Loop:

To run the test loop, simply call the `check_accuracy` function with your test data loader and trained model after each epoch of training.

```python
# Assuming you have a test_loader and a trained model called model
for epoch in range(num_epochs):
    # Training loop (not shown here)
    train_model(model, train_loader, optimizer, criterion)

    # Evaluation on test dataset
    check_accuracy(test_loader, model)
```
