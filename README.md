# Handwritten Digit Recognizer

A machine learning project that recognizes handwritten digits (0–9) using a neural network trained on the MNIST dataset.

## Overview

This project explores the implementation and training of a neural network for handwritten digit classification. It uses 28×28 grayscale images containing handwritten digits and learns to identify the digit represented by each image.

The goal is to understand the underlying mechanics of neural networks, from processing raw pixel data to making predictions.

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- Total images: 70,000
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Input features: 784 pixel values
- Classification classes: 10 (digits 0–9)
- Pixel intensity range: 0–255

Each image is represented by 784 pixel intensity values, accompanied by its corresponding digit label.

## Neural Network Architecture

The network uses a fully connected feedforward architecture:

```text
Input Layer:    784 neurons
Hidden Layer 1: 196 neurons
Hidden Layer 2: 98 neurons
Hidden Layer 3: 32 neurons
Output Layer:   10 neurons
```

The output layer produces predictions across ten possible digit classes.

## Key Concepts

- Neural network architecture
- Forward propagation
- Activation functions
- Loss calculation
- Backpropagation
- Gradient descent
- Weight and bias optimization
- Model evaluation

## Project Structure

```text
handwritten-digit-recognizer/
│
├── dataset/
│   └── mnist_dataset.csv
│
├── src/
│
├── README.md
└── requirements.txt
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/sparshpatil13412/handwritten-digits-recognition-using-numpy
cd handwritten-digit-recognizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Future Improvements

- Experiment with different network architectures.
- Evaluate the impact of different activation functions.
- Analyze model performance and classification errors.
- Explore optimization techniques.

## Acknowledgements

Dataset: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

**Status:** In Development