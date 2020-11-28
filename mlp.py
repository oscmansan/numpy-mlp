import os
import urllib.request

import numpy as np
from tqdm import trange


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse(a, y):
    return 0.5 * np.sum((y - a) ** 2)


def mse_grad(a, y):
    return a - y


class MLP:

    def __init__(self, sizes):
        self.sizes = sizes
        
        self.weights = [np.random.randn(y, x).astype(np.float32) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1).astype(np.float32) for y in self.sizes[1:]]
    
    def forward(self, x, intermediate_outputs=False):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ x + b
            x = sigmoid(z)
            if intermediate_outputs:
                zs.append(z)
                activations.append(x)
        if intermediate_outputs:
            return activations, zs
        else:
            return x
    
    def backprop(self, x, y):
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        grad_biases = [np.zeros(b.shape) for b in self.biases]

        # forward
        activations, zs = self.forward(x, intermediate_outputs=True)
        loss = mse(activations[-1], y)  # TODO: use cross-entropy

        # backward
        delta = mse_grad(activations[-1], y) * sigmoid_grad(zs[-1])
        grad_weights[-1] = delta @ activations[-2].T
        grad_biases[-1] = delta
        for l in range(2, self.num_layers):
            delta = (self.weights[-l+1].T @ delta) * sigmoid_grad(zs[-l])
            grad_weights[-l] = delta @ activations[-l-1].T
            grad_biases[-l] = delta
        
        return loss, grad_weights, grad_biases
    
    def fit(self, train_data, epochs, batch_size, lr, val_data=None):
        log = {}
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            with trange(0, len(train_data), batch_size, desc=f'Epoch {epoch}', leave=(epoch == epochs-1)) as t:
                for i in t:
                    batch = train_data[i:i+batch_size]
                    loss = self.train_step(batch, lr)
                    log['train_loss'] = f'{loss:.6f}'
                    if i % 1000 == 0:
                        t.set_postfix(**log)
                if val_data:
                    accuracies = []
                    for i in trange(0, len(val_data), batch_size, desc='Validating', leave=False):
                        batch = val_data[i:i+batch_size]
                        accuracy = self.val_step(batch)
                        accuracies.append(accuracy)
                    log['val_acc'] = f'{np.mean(accuracies):05.2f}'
    
    def train_step(self, batch, lr):
        sum_grad_weights = [np.zeros(w.shape) for w in self.weights]
        sum_grad_biases = [np.zeros(b.shape) for b in self.biases]
        losses = []
        for x, y in batch:  # TODO: compute all batch elements at once
            loss, grad_weights, grad_biases = self.backprop(x, y)
            losses.append(loss)
            sum_grad_weights = map(sum, zip(sum_grad_weights, grad_weights))
            sum_grad_biases = map(sum, zip(sum_grad_biases, grad_biases))
        self.weights = [w - lr * gw / len(batch) for w, gw in zip(self.weights, sum_grad_weights)]
        self.biases = [b - lr * gb / len(batch) for b, gb in zip(self.biases, sum_grad_biases)]
        return np.mean(losses)
    
    def val_step(self, batch):
        accuracy = sum([int(np.argmax(self.forward(x)) == y) for x, y in batch]) / len(batch) * 100.0
        return accuracy

    @property
    def num_layers(self):
        return len(self.sizes)


if __name__ == "__main__":
    filename = 'mnist.npz'
    if not os.path.exists(filename):
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', filename)
    with np.load(filename) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']
    
    train_examples = train_examples.reshape((train_examples.shape[0], -1, 1)).astype(np.float32) / 255.
    test_examples = test_examples.reshape((test_examples.shape[0], -1, 1)).astype(np.float32) / 255.
    train_targets = np.zeros((train_labels.shape[0], 10, 1), dtype=np.float32)
    train_targets[np.arange(train_labels.shape[0]), train_labels] = 1.
    test_labels = test_labels.astype(int)
    
    train_data = list(zip(train_examples, train_targets))
    test_data = list(zip(test_examples, test_labels))

    model = MLP(sizes=[784, 30, 10])

    model.fit(train_data, epochs=30, batch_size=10, lr=3.0, val_data=test_data)
