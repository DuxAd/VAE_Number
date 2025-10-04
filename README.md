# Convolutional VAE on MNIST: Latent Space Generation and Analysis

## Introduction

This project implements a **Convolutional Variational Autoencoder (C-VAE)** using the MNIST handwritten digits dataset.

The primary objectives were to:
1.  **Reduce the dimensionality** of the $28 \times 28 \times 1$ images to a  **compact latent space ($Z_{dim}=3$)**.
2.  Demonstrate the continuity of this latent space, enabling the generation of new digits and interpolation between classes.

---

## Model Architecture

The model is defined within the Autoencoder class and uses a classic symmetric architecture:

* **Encoder :** Uses Conv2D layers and LeakyReLU activations to map input images to the parameters of the latent distribution ($\mu$ and  $\log(\sigma^2)$ ).
* **Decoder :** Uses Conv2DTranspose layers to reconstruct the image from a point in the latent space $Z$.
* **VAE Loss :** The cost function combines the Reconstruction Loss (Binary Cross-Entropy) and the KL Regularization Loss (to enforce a standard normal distribution on the latent space).

---

## Key Results

The results validate a successful training and meaningful latent representation:

1.  **High-Quality Reconstruction:** The model accurately reconstructs input digits with a very low BCE loss (approx. 0.02).
2.  **Coherent Random Generation:** Sampling random points from the latent space produces clear, recognizable, and novel digit forms.
3.  **Organized Latent Space:** 3D visualizations show that digit classes form distinct yet smoothly connected clusters, which facilitates accurate interpolation.
4.  **Fluid Morphing:** Linear interpolation between latent clusters (e.g., 8 to 1) results in smooth and realistic visual transitions.
