import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
np.set_printoptions(suppress = True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x <= 0, 0, 1)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1-np.tanh(x)**2

class Net:
    def __init__(self, X, Y, method, method_deriv, epoch=1000, learning_rate=0.1, stop=0.0, momentum=False, momentum_rate=0.0):

        self.input = X
        self.output = Y
        self.method = method
        self.method_deriv = method_deriv
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.stop = stop
        self.momentum = momentum

        self.w1 = np.random.randn(2, 2)
        self.w2 = np.random.randn(1, 2)

        self.b1 = np.ones((2, 1))
        self.b2 = np.ones((1, 1))

        if self.momentum:
            self.momentum_w1 = np.zeros((2, 2))
            self.momentum_w2 = np.zeros((1, 2))

            self.momentum_b1 = np.zeros((2, 1))
            self.momentum_b2 = np.zeros((1, 1))
            self.momentum_rate = momentum_rate

        self.a1 = None
        self.a2 = None

    def forward(self, x):
        self.a1 = self.method(np.dot(self.w1, x) + self.b1)
        self.a2 = self.method(np.dot(self.w2, self.a1) + self.b2)
        return self.a2

    def backprop(self, error):
        m = self.input.shape[1]

        delta2 = self.a2 - self.output
        w2 = np.dot(delta2, self.a1.T) / m
        b2 = np.sum(delta2, axis=1, keepdims=True)

        delta1 = np.multiply(np.dot(self.w2.T, delta2), self.method_deriv(self.a1))
        w1 = np.dot(delta1, self.input.T) / m
        b1 = np.sum(delta1, axis=1, keepdims=True) / m

        self.updateParams(w1, w2, b1, b2)

    def updateParams(self, w1, w2, b1, b2):
        if self.momentum:
            self.momentum_w1 = self.momentum_rate * self.momentum_w1 + self.learning_rate * w1
            self.momentum_w2 = self.momentum_rate * self.momentum_w2 + self.learning_rate * w2
            self.momentum_b1 = self.momentum_rate * self.momentum_b1 + self.learning_rate * b1
            self.momentum_b2 = self.momentum_rate * self.momentum_b2 + self.learning_rate * b2

            self.w1 -= self.momentum_w1
            self.w2 -= self.momentum_w2
            self.b1 -= self.momentum_b1
            self.b2 -= self.momentum_b2
        else:
            self.w1 -= self.learning_rate * w1
            self.w2 -= self.learning_rate * w2
            self.b1 -= self.learning_rate * b1
            self.b2 -= self.learning_rate * b2

    def train(self):
        self.errors = []
        for i, epoch in enumerate(range(self.epoch)):
            self.forward(self.input)
            error = self.a2 - self.output
            self.backprop(error)
            self.errors.append(np.mean(np.abs(error)))

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} Error: {np.mean(np.abs(error))}")

            if np.mean(np.abs(error)) <= self.stop:
                print('Achieved stop condition')
                print(f"Epoch: {epoch} Error: {np.mean(np.abs(error))}")
                break

        if self.stop:
            stop = np.linspace(self.stop, self.stop, i + 1)
            plt.plot(list(range(len(self.errors))), stop, label=f"Stop Condition = {self.stop}", color='red')

        plt.plot(list(range(len(self.errors))), self.errors, label="Errors")
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Mean absolute error')
        plt.show()

    def predict(self, X):
        threshold = 0.55
        predictions = self.forward(X)
        print()
        print(" X     score   result")
        for i, prediction in enumerate(predictions[0]):
            print(f'{X[0][i]} {X[1][i]} |  {round(prediction, 3)}{" " * (5 - len(str(round(prediction, 3))))} |  {(prediction >= threshold) * 1}')


X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
X_test = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])


net = Net(X, Y, tanh, tanh_deriv, epoch=5000, learning_rate=0.1, stop=0, momentum=False, momentum_rate=0.0)
net.train()
net.predict(X_test)