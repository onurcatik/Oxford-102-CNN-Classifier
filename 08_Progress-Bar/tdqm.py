from tqdm import tqdm
import pandas as pd
import numpy as np
import time

# Example 1: Loading Data
print("Example 1: Loading Data")
data = range(100)
for _ in tqdm(data, desc="Loading"):
    time.sleep(0.05)  # Simulating data loading

# Example 2: Data Transformation
print("\nExample 2: Data Transformation")
data = range(100)
transformed_data = []
for item in tqdm(data, desc="Transforming"):
    transformed_data.append(item * 2)
    time.sleep(0.05)  # Simulating data transformation

# Example 3: Image Processing
print("\nExample 3: Image Processing")
image = np.random.rand(100, 100)  # Random image data
processed_image = np.zeros_like(image)
for i in tqdm(range(image.shape[0]), desc="Processing Rows"):
    for j in range(image.shape[1]):
        processed_image[i, j] = image[i, j] * 2
        time.sleep(0.0001)  # Simulating image processing

# Example 4: Simulating Experiment Trials
print("\nExample 4: Simulating Experiment Trials")
num_trials = 50
for trial in tqdm(range(num_trials), desc="Trials"):
    # Simulating trial process
    time.sleep(0.1)

# Example 5: Text Processing
print("\nExample 5: Text Processing")
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
processed_text = ""
for char in tqdm(text, desc="Processing Characters"):
    if char.isalpha():
        processed_text += char.upper()
    else:
        processed_text += char
    time.sleep(0.05)  # Simulating text processing

# Example 6: Parallel Processing
print("\nExample 6: Parallel Processing")
tasks = 20
for _ in tqdm(range(tasks), desc="Tasks", position=0):
    # Simulating parallel tasks
    time.sleep(0.2)

# Example 7: Machine Learning Training
print("\nExample 7: Machine Learning Training")
num_epochs = 5
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    # Simulating training process
    time.sleep(1)

# Example 8: Custom Progress Bar Style
print("\nExample 8: Custom Progress Bar Style")
for i in tqdm(range(100), desc="Custom Style", bar_format="{desc}: {percentage:.0f}%|{bar}|"):
    time.sleep(0.05)  # Simulating computation

print("\nAll examples completed.")
