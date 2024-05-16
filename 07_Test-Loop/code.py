import torch

def check_accuracy(loader, model, device):
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

# Example usage
# Assuming you have defined your model, optimizer, criterion, train_loader, and test_loader

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming you have a trained model called model
# Assuming you have defined num_epochs and optimizer elsewhere
for epoch in range(num_epochs):
    # Training loop (not shown here)
    train_model(model, train_loader, optimizer, criterion)

    # Evaluation on test dataset
    check_accuracy(test_loader, model, device)
