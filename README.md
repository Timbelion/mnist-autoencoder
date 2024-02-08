# mnist-autoencoder

Simple convolutional autoencoder with only two latent neurons in the hidden layer. Trained on mnist dataset of 28x28 images of handwritten digits, where network has to predict the same image it was fed with.

# Requirements

Module  | Version
------------- | -------------
keras  | 2.9.0
tensorflow  | 1.1.0
numpy  | 1.21.6
matplotlib  | 3.2.2

# Instructions

Simply run *main.py* and wait for the AI to train itself. After running *main.py* for the first time it will generate two files: *latent_space_image.json* and *trained_model_cnn.h5*. Once you have this files generated you can comment out unnecessary lines in *main.py* (see file *main.py* for more clarification).

# Video Showcase

https://github.com/Timbelion/mnist-autoencoder/assets/76007113/8eca9452-4446-4299-8af8-fc4ec652b163



