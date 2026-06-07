import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file properly
file_path = r"C:\Users\lion7\Documents\GitHub\Computer_Vision_Project\src\mnist_test.csv"

with open(file_path, 'r') as f:
    # Read the first line (or any specific line you want)
    line = f.readline().strip()   # Change to f.readlines()[n] if you want the nth image

# Split by comma
values = line.split(',')

# First value is usually the label, the rest are 784 pixels
label = int(values[0])
pixels = values[1:]  # 784 pixel values

# Convert to numpy array and reshape to 28x28
img_array = np.array(pixels, dtype=int).reshape(28, 28)

print(f"Label: {label}")
print(f"Image shape: {img_array.shape}")

# === Plot the heatmap ===
plt.figure(figsize=(8, 8))
plt.imshow(img_array, cmap='gray_r', interpolation='nearest')  # gray_r = white background

plt.title(f'28x28 MNIST Digit (Label: {label})')
plt.colorbar(label='Pixel Intensity (0 = Black, 255 = White)')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
