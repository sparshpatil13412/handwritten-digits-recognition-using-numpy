import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
y_train = np.array(y_train).T
y_val = np.array(y_val).T

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
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A

#creating loss function
def loss(y, y_hat):
    cost = np.sum(y * np.log(y_hat))
    return cost

#creaing the model
class DigitClassifier():
    def __init__(self, X, y, layer1, layer2, learning_rate, epochs):
        self.W1 = np.random.randn(784, layer1) * 0.01
        self.b1 = np.zeros((1, layer1))
        self.W2 = np.random.randn(layer1, layer2) * 0.01
        self.b2 = np.zeros((1, layer2))
        self.W3 = np.random.randn(layer2, 10) * 0.01
        self.b3 = np.zeros((1, 10))
        self.lr = learning_rate
        self.X = X
        self.y = y
        self.epochs = epochs
        self.m = X.shape[0]
        self.costs = []
    
    def forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = ReLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = tanh(self.Z2)

        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        self.A3 = softmax(self.Z3)

        cost = loss(self.y, self.A3)

        return cost

    def backward(self):
        self.dZ3 = self.A3 - self.y
        self.dW3 = (1 / self.m) * np.dot(self.dZ3, self.A2.T)
        self.db3 = (1 / self.m) * np.sum(self.dZ3, axis=1, keepdims=True)

        self.dA2 = np.dot(self.W3.T, self.dZ3)
        self.dZ2 = self.dA2 * (1 - self.A2 ** 2)
        self.dW2 = (1 / self.m) * np.dot(self.dZ2, self.A1.T)
        self.db2 = (1 / self.m) * np.sum(self.dZ2)

        self.dA1 = np.dot(self.W2.T, self.dZ2)
        self.dZ1 = self.dA1 * (self.Z1 > 0)
        self.dW1 = (1 / self.m) * np.dot(self.dZ1, self.X.T)
        self.db1 = (1 / self.m) * np.sum(self.dZ1)

    def descent(self):
        self.W3 -= self.lr * self.dW3
        self.b3 -= self.lr * self.db3
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1

    def train(self):
        for epoch in range(self.epochs):
            cost = self.forward()
            self.costs.append(cost)
            self.backward()
            self.descent()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")
    
    def visualize(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.costs, label="Training Cost", color="#1f77b4", linewidth=2)
        plt.title(f"Cost vs. Epochs ({self.epochs} Epochs)", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Cost", fontsize=12)
        plt.xlim(1, self.epochs)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = ReLU(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = ReLU(Z2)

        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = softmax(Z3)

        confidence = A3
        predictions = np.argmax(A3, axis=0)
        return confidence, predictions