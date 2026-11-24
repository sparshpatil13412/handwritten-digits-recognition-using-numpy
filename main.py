import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

#setting random seed
np.random.seed(42)

#loading the data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "train.csv"
train = pd.read_csv(train_path)
print(train.head(2))

test_path = DATA_DIR / "test.csv"
test = pd.read_csv(test_path)
print(test.head(2))

#convert data into numpy arrays
X = train.drop(columns=["label"]).values
y = train["label"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train).T
X_val = np.array(X_val).T
y_train = np.eye(10)[y_train].T
y_val = np.eye(10)[y_val].T

print("X train shape:", X_train.shape)
print("X validation shape:", X_val.shape)
print("y train shape:", y_train.shape)
print("y validation shape:", y_val.shape)

#creating activation functions and their derivatives
def ReLU(z):
    A = np.maximum(z, 0)
    return A

def tanh(z):
    A = np.tanh(z)
    return A

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z)
    return A / np.sum(A, axis=0, keepdims=True)

#creating loss function
def loss(y, y_hat):
    m = y.shape[1]
    cost = -(1 / m) * np.sum(y * np.log(y_hat + 1e-8))
    return cost

#creating the model class
class DigitClassifier():
    def __init__(self, X, y, layer1, layer2, learning_rate, epochs, batch_size):
        self.W1 = np.random.randn(784, layer1) * np.sqrt(2 / 784)
        self.b1 = np.zeros((layer1, 1 ))
        self.W2 = np.random.randn(layer1, layer2) * np.sqrt(2 / (layer1 + layer2))
        self.b2 = np.zeros((layer2, 1))
        self.W3 = np.random.randn(layer2, 10) * np.sqrt(2 / layer2)
        self.b3 = np.zeros((10, 1))
        self.X = X
        self.y = y
        self.lr = learning_rate
        self.epochs = epochs
        self.costs = []
        self.batch_size = batch_size
    
    def forward(self, X, y):
        self.Z1 = np.dot(self.W1.T, X) + self.b1
        self.A1 = ReLU(self.Z1)

        self.Z2 = np.dot(self.W2.T, self.A1) + self.b2
        self.A2 = tanh(self.Z2)

        self.Z3 = np.dot(self.W3.T, self.A2) + self.b3
        self.A3 = softmax(self.Z3)

        cost = loss(y, self.A3)

        return cost

    def backward(self, X, y):
        m = X.shape[1]
        self.dZ3 = self.A3 - y
        self.dW3 = (1 / m) * np.dot(self.A2, self.dZ3.T)
        self.db3 = (1 / m) * np.sum(self.dZ3, axis=1, keepdims=True)

        self.dA2 = np.dot(self.W3, self.dZ3)
        self.dZ2 = self.dA2 * (1 - self.A2 ** 2)
        self.dW2 = (1 / m) * np.dot(self.A1, self.dZ2.T)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)

        self.dA1 = np.dot(self.W2, self.dZ2)
        self.dZ1 = self.dA1 * (self.Z1 > 0)
        self.dW1 = (1 / m) * np.dot(X, self.dZ1.T)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def descent(self):
        self.W3 -= self.lr * self.dW3
        self.b3 -= self.lr * self.db3
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1

    def train(self):
        for epoch in range(1, self.epochs + 1):
            #apply mini-batch sgd
            indices = np.random.permutation(len(self.X))
            X_shuffled = self.X[:, indices]
            y_shuffled = self.y[:, indices]
            epoch_cost = 0
            for start in range(0, len(self.X), self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                cost = self.forward(X_batch, y_batch)
                epoch_cost += cost * X_batch.shape[1]
                self.backward(X_batch, y_batch)
                self.descent()

            epoch_cost /= X.shape[1]
            self.costs.append(epoch_cost)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch} / {self.epochs}, Cost: {epoch_cost :.4f}")
    
    def visualize(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.costs, label="Training Cost", color="#1f77b4", linewidth=2)
        plt.title(f"Cost vs. Epochs ({self.epochs} Epochs)", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Cost", fontsize=12)
        plt.xlim(0, self.epochs)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        Z1 = np.dot(self.W1.T, X) + self.b1
        A1 = ReLU(Z1)

        Z2 = np.dot(self.W2.T, A1) + self.b2
        A2 = tanh(Z2)

        Z3 = np.dot(self.W3.T, A2) + self.b3
        A3 = softmax(Z3)

        confidence = A3.reshape(A3.shape[0])
        prediction = np.argmax(A3, axis=0)
        return confidence, prediction

#defining the model
layer1 = 196
layer2 = 32
learning_rate = 0.001
epochs = 300
batch_size = 32
model = DigitClassifier(X_train, y_train, layer1, layer2, learning_rate, epochs, batch_size)

#training the model, visualizing cost and recording the train and validation accuracy
model.train()
model.visualize()

correct = 0
total = X_train.shape[0]
for i in range(total):
    X_sample = X_train[:, i].reshape(784, 1)

    confidence, prediction = model.predict(X_sample)

    actual = np.argmax(y_train[:, i])

    if prediction[0] == actual:
        correct += 1

accuracy = (correct / total) * 100
print(f"Train accuracy is {accuracy :.0f}%")

correct = 0
total = X_val.shape[0]
for i in range(total):
    X_sample = X_val[:, i].reshape(784, 1)

    confidence, prediction = model.predict(X_sample)

    actual = np.argmax(y_val[:, i])

    if prediction[0] == actual:
        correct += 1

accuracy = (correct / total) * 100
print(f"Validation accuracy is {accuracy :.0f}%")

#Predicting on test examples
image1 = Image.open("image1.png").convert("L")  # Convert to grayscale
image2 = Image.open("image2.png").convert("L")  # Convert to grayscale
image3 = Image.open("image3.png").convert("L")  # Convert to grayscale

def resize_and_convert_into_array(image):
    image = image.resize((28, 28))
    image.show()
    image = np.array(image)
    image = image.reshape(784, 1)
    return image

image1 = resize_and_convert_into_array(image1)
image2 = resize_and_convert_into_array(image2)
image3 = resize_and_convert_into_array(image3)

confidence1, prediction1 = model.predict(image1)
print(f"The model has predicted {prediction1} for image 1 with confidence {np.max(confidence1 * 100) :.0f}%")
confidence2, prediction2 = model.predict(image2)
print(f"The model has predicted {prediction2} for image 2 with confidence {np.max(confidence2 * 100) :.0f}%")
confidence3, prediction3 = model.predict(image3)
print(f"The model has predicted {prediction3} for image 3 with confidence {np.max(confidence3 * 100) :.0f}%")