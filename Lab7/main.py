import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function for Cellular Automata (Edge Detection or Noise Reduction)
def cellular_automata(image, iterations=10, threshold=30):
    grid = image.copy()  # Initialize grid (image as 2D array)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for iteration in range(iterations):
        updated_grid = grid.copy()
        
        for i in range(1, len(grid) - 1):  # Loop through pixels (excluding borders)
            for j in range(1, len(grid[0]) - 1):
                pixel = grid[i, j]
                neighbor_vals = [grid[i+di, j+dj] for (di, dj) in neighbors]
                
                # Edge detection: large difference with neighbors indicates edge
                if max(neighbor_vals) - min(neighbor_vals) > threshold:
                    updated_grid[i, j] = 255  # Edge pixel
                else:
                    # Noise reduction: average with neighbors for smoothing
                    new_pixel_value = sum(np.clip(neighbor_vals, 0, 255)) // 8  # Clipping before averaging
                    
                    # Clip the new pixel value to the range 0-255
                    updated_grid[i, j] = np.clip(new_pixel_value, 0, 255)
                
        grid = updated_grid  # Update the grid with new values
    
    return grid  # Output updated image

# Set numpy to ignore overflow warnings
np.seterr(over='ignore')

# Generate a smaller dummy grayscale image (random noise)
# Create a 5x5 pixel image with random values between 0 and 255
image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)

# Print the original image
print("Original Image (Pixel Values):")
for row in image:
    print(row)

# Apply the cellular automata algorithm
iterations = 10
threshold = 30
processed_image = cellular_automata(image, iterations, threshold)

# Print the processed image
print("\nProcessed Image (Pixel Values):")
for row in processed_image:
    print(row)

# Visualize the images using matplotlib
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Processed Image')
plt.imshow(processed_image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()
