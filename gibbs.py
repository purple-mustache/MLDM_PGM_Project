# Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method
# that sequentially samples from the conditional distribution 
# of each variable given the current values of all other variables. For image sampling
# , you would sample each pixel value given the values of all other pixels. 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def gibbs_sampling(original_img, iterations=10):
    current_img = original_img.copy()

    for _ in range(iterations):
        for i in range(current_img.shape[0]):
            for j in range(current_img.shape[1]):
                # Sample pixel (i, j) using conditional distribution
                current_img[i, j] = sample_pixel_conditionally(current_img, i, j)

    return current_img

# Function to sample a pixel conditionally in a Gibbs sampling step
def sample_pixel_conditionally(img, i, j):
    # Modify this function based on the conditional distribution of the pixel
    # In this example, we add Gaussian noise to the current pixel value
    return np.random.normal(img[i, j], 5)


# Example usage
image_path = f'{data_path}/30.jpg'  # Replace with the path to your image
image = read_image(image_path)

# num_samples = 1000
samples = gibbs_sampling(image)

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(samples, cmap='gray')
plt.title('Generated Image (Gibbs Sampling)')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(image - samples), cmap='gray')
plt.title('Pixel-wise Difference')

plt.show()








