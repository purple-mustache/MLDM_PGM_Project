import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    return image

def metropolis_hastings(image, num_samples, proposal_std):
    samples = [np.copy(image)]
    image_shape = image.shape

    for _ in range(num_samples):
        # Propose a new image by adding random noise
        proposed_image = samples[-1] + np.random.normal(scale=proposal_std, size=image_shape)

        # Evaluate the likelihood of the proposed image
        likelihood_proposed = likelihood_function(proposed_image)

        # Evaluate the likelihood of the current image
        likelihood_current = likelihood_function(samples[-1])

        # Compute acceptance ratio
        acceptance_ratio = min(1, likelihood_proposed / likelihood_current)

        # Accept or reject the proposed image
        if np.random.uniform() < acceptance_ratio:
            samples.append(proposed_image)
        else:
            samples.append(samples[-1])

    return np.array(samples)

def likelihood_function(image):
    # Example: Compute the negative sum of squared differences from a target image
    target_image = np.zeros_like(image)  # Replace with your target image
    return -np.sum((image - target_image)**2)


image_path = f'{data_path}/30.jpg'  
initial_image = read_image(image_path)

num_samples = 10
proposal_std = 0.1

samples = metropolis_hastings(initial_image, num_samples=num_samples, proposal_std=proposal_std)

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(initial_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
# Assuming 'samples' is a list of generated images
for sample in samples:
    plt.imshow(sample, cmap='gray')
plt.title('Generated Image (MH Sampling)')

plt.subplot(1, 3, 3)
# Assuming 'samples' is a list of generated images
for sample in samples:
    plt.imshow(np.abs(initial_image - sample), cmap='gray')
plt.title('Pixel-wise Difference')

plt.show()
