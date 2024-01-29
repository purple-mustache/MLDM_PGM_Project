# Hamiltonian Monte Carlo (HMC) is a powerful MCMC method that 
# uses Hamiltonian dynamics to propose new states in the parameter space. 
# It is particularly useful when sampling from high-dimensional spaces. 
# Sampling from an image distribution using HMC involves defining a target 
# distribution that represents the likelihood of the image given some parameters.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# Set a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load a real image (you can replace this with your image loading logic)
# Example: Load an image of your choice
image = plt.imread('0.22__30/30.jpg')
image = image[:, :, 0]  # Use only one channel (grayscale)

# Flatten the image to use pixel intensities as data points
data = image.flatten().astype(np.float32)

# Function to visualize images
def plot_images(images, titles, cmap='gray'):
    plt.figure(figsize=(12, 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Visualize the original image
plot_images([image], ['Original Image'])

# Define the Hamiltonian Monte Carlo sampling function
def hmc_sample(initial_params, observed_data, num_samples, step_size=0.01, num_leapfrog_steps=10):
    dtype = tf.float32
    
    # Define a Gaussian Mixture Model (GMM) as the likelihood
    mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.Normal(loc=initial_params[0], scale=initial_params[1])
    )
    
    # Define a simple prior for the parameters
    prior = tfd.Independent(tfd.Normal(loc=[0.0, 0.0], scale=[10.0, 10.0]), reinterpreted_batch_ndims=1)
    
    # Define the joint distribution (product of prior and likelihood)
    target_log_prob_fn = lambda *args: prior.log_prob(args) + mixture.log_prob(observed_data)
    
    # Initialize the Hamiltonian Monte Carlo kernel
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps
    )

    # Run the chain
    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=100,
        current_state=initial_params,
        kernel=hmc
    )

    return samples

# Initial guess for parameters
initial_params = tf.Variable([0.0, 1.0], dtype=tf.float32)

# Run HMC sampling
num_samples = 500
samples = hmc_sample(initial_params, data, num_samples)

# Visualize the sampled images
sampled_images = [np.reshape(sampled_params, image.shape) for sampled_params in samples.numpy()]
plot_images(sampled_images, ['Sampled Image {}'.format(i+1) for i in range(num_samples)])
