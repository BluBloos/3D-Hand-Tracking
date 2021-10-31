import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

np.set_printoptions(threshold=np.inf)


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((784, 10)) - 1
        self.biases = np.zeros((10,))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_ouputs, training_iteration):
        training_ouput_nodes = self.training_output_nodes(training_ouputs)
        for iteration in range(training_iteration):
            output = self.think(training_inputs)

            error = (training_ouput_nodes - output)
            adjustments_w = np.dot(training_inputs.T, error * self.sigmoid_derivative(output) / np.size(training_ouputs))
            adjustments_b = error.mean(0)

            self.weights = self.weights + adjustments_w
            self.biases = self.biases + adjustments_b


    def think(self, inputs):

        output = self.sigmoid(np.dot(inputs, self.weights))
        output += self.biases
        return output

    def training_output_nodes(self, training_outputs):
        transformation_array = np.array([[1 / 11],
                                         [1],
                                         [1 / 2],
                                         [1 / 3],
                                         [1 / 4],
                                         [1 / 5],
                                         [1 / 6],
                                         [1 / 7],
                                         [1 / 8],
                                         [1 / 9]])
        training_outputs = np.expand_dims(training_outputs, axis=1)
        indicator = np.dot(training_outputs, transformation_array.T)

        indicator[np.all((indicator == 0), axis=1)] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        indicator[indicator != 1] = 0

        return indicator


if __name__ == "__main__":
    neural_network = NeuralNetwork()
    weights1 = neural_network.weights
    # print("Random weights before training: ")
    # print(neural_network.weights)

    training_inputs, training_outputs = mnist["data"], mnist["target"]
    training_outputs = training_outputs.astype(float)
    training_inputs = training_inputs / 255

    neural_network.train(training_inputs, training_outputs, 20000)
    print("maximum weights: ")
    print(np.max(neural_network.weights))
    print("minimum weights: ")
    print(np.min(neural_network.weights))
    print("maximum biases: ")
    print(np.max(neural_network.biases))
    print("mininum biases: ")
    print(np.min(neural_network.biases))


