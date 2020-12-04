import os
import argparse
import urllib.request

import numpy as np
from tqdm import tqdm  # for the progress bars


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    # exploits the fact that the derivative is a simple function of the output from the sigmoid function
    return z * (1 - z)


def softmax(x):
    x = np.exp(x - np.amax(x, axis=1, keepdims=True))
    x /= np.sum(x, axis=1, keepdims=True)
    return x


def mse(x, y):
    return ((x - y) ** 2).mean(axis=1) / 2


def cross_entropy(x, y, eps=1e-15):
    x = np.clip(x, eps, 1 - eps)
    x /= x.sum(axis=1)[:, np.newaxis]
    return -(y * np.log(x)).sum(axis=1)


def data_loader(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        x, y = zip(*batch)
        x = np.array(x)
        y = np.array(y)
        yield x, y


class MLP:

    def __init__(self, sizes):
        self.sizes = sizes
        
        # initialize params
        self.weights = [np.random.randn(y, x).astype(np.float32) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y).astype(np.float32) for y in self.sizes[1:]]
    
    def forward(self, x):
        activations = [x]

        # iterate over hidden layers
        for i in range(self.num_layers - 1):
            x = x @ self.weights[i].T + self.biases[i]
            if i < (self.num_layers - 2):
                x = sigmoid(x)  # for the hidden layers
            else:
                x = softmax(x)  # for the last layer
            activations.append(x)

        return activations

    def backprop(self, x, y):
        bsz = x.shape[0]

        # forward propagate
        activations = self.forward(x)
        loss = cross_entropy(activations[-1], y)

        # backward propagate
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # compute gradient for the last layer
        delta = activations[-1] - y
        grad_weights[-1] = delta.reshape((bsz, -1, 1)) @ activations[-2].reshape((bsz, 1, -1))
        grad_biases[-1] = delta

        # iterate over hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            delta = (delta @ self.weights[i]) * sigmoid_derivative(activations[i])
            grad_weights[i - 1] = delta.reshape((bsz, -1, 1)) @ activations[i - 1].reshape((bsz, 1, -1))
            grad_biases[i - 1] = delta
        
        return loss, grad_weights, grad_biases
    
    def fit(self, train_data, epochs, batch_size, lr, val_data=None):
        log = {}
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            with tqdm(data_loader(train_data, batch_size), desc=f'Epoch {epoch}', leave=(epoch == epochs-1)) as pbar:
                for batch in pbar:
                    loss = self.train_step(batch, lr)
                    log['train_loss'] = f'{loss:.6f}'
                    pbar.set_postfix(**log)
                if val_data:
                    accuracies = []
                    for batch in tqdm(data_loader(val_data, batch_size), desc='Validating', leave=False):
                        accuracy = self.val_step(batch)
                        accuracies.append(accuracy)
                    log['val_acc'] = f'{np.mean(accuracies):05.2f}'
    
    def train_step(self, batch, lr):
        x, y = batch

        # compute gradients
        loss, grad_weights, grad_biases = self.backprop(x, y)
        grad_weights = [gw.mean(axis=0) for gw in grad_weights]
        grad_biases = [gb.mean(axis=0)for gb in grad_biases]

        # update params
        self.weights = [w - lr * gw for w, gw in zip(self.weights, grad_weights)]
        self.biases = [b - lr * gb for b, gb in zip(self.biases, grad_biases)]

        return loss.mean(axis=0)
    
    def val_step(self, batch):
        x, y = batch
        bsz = x.shape[0]

        # compute validation accuracy
        logits = self.forward(x)[-1]
        predicted = logits.argmax(axis=1)
        correct = (predicted == y).sum()
        accuracy = 100 * correct / bsz

        return accuracy

    @property
    def num_layers(self):
        return len(self.sizes)


def to_tensor(x):
    return x.reshape((x.shape[0], -1)).astype(np.float32) / 255.


def one_hot(y, num_classes=10):
    z = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    z[np.arange(y.shape[0]), y] = 1.
    return z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[100])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    args = parser.parse_args()

    # load mnist dataset
    filename = os.path.join(args.data_dir, 'mnist.npz')
    if not os.path.exists(filename):
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', filename)
    with np.load(filename) as data:
        train_samples = data['x_train']
        train_labels = data['y_train']
        test_samples = data['x_test']
        test_labels = data['y_test']
    
    num_features = np.prod(train_samples.shape[1:])
    num_classes = np.unique(train_labels).size

    # preprocess data
    train_samples = to_tensor(train_samples)
    test_samples = to_tensor(test_samples)
    train_targets = one_hot(train_labels, num_classes)
    test_labels = test_labels.astype(int)
    
    train_data = list(zip(train_samples, train_targets))
    test_data = list(zip(test_samples, test_labels))

    # train mlp on mnist
    model = MLP(sizes=[num_features, *args.hidden_dims, num_classes])
    model.fit(train_data, args.epochs, args.batch_size, args.learning_rate, val_data=test_data)
