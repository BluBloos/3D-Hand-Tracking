import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784')




class NeuralNetwork():

    def __init__(self):#initialize weights and biases for the neural network
        np.random.seed(1)
        self.weights = 2 * np.random.random((784, 10)) - 1
        self.biases = np.zeros((10,))
    def sigmoid(self, x): #sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):#derivative of the sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_ouputs, training_iteration): #training the system a certain amount of times
        training_ouput_nodes = self.training_output_nodes(training_ouputs)
        for iteration in range(training_iteration):
            output = self.think(training_inputs) #neural net will output 10 node values for each input image (60,000x10 matrix)

            error = (training_ouput_nodes - output)     #60,000x10 matrix that contains the offset of every output node vs the expected  output
            adjustments_w = np.dot(training_inputs.T, error * self.sigmoid_derivative(output) / np.size(training_ouputs))   #calculating weights adjustments using the gradient
            adjustments_b = error.mean(0)   #bias adjustment for only one layer is just the average offset of each output node vs the expected output
            #we expect the biases for a 1 layer neural net to be zero

            self.weights = self.weights + adjustments_w
            self.biases = self.biases + adjustments_b
            #weight and bias adjustments
        print("Model accuracy: ")
        print(neural_network.test_accuracy(testing_inputs,testing_outputs))     #print the model accuracy

    def think(self, inputs):    #This function returns the output of the 10 nodes based on the input images passed in

        output = self.sigmoid(np.dot(inputs, self.weights))  #matrix multiplication to get the output
        output += self.biases   #adding on the biases
        return output

    def training_output_nodes(self, training_outputs):  #function that does one-hot-encoding on the expected output
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
        indicator = np.dot(training_outputs, transformation_array.T)     #will leave a 1 in the correct spot for each entry

        indicator[np.all((indicator == 0), axis=1)] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]     #every all-zero row set to [1,0,0,0,0,0,0,0,0,0]
        indicator[indicator != 1] = 0   #everything not a 1 set to zero

        return indicator    #returns the one-hot-encoded expected output

    def test_accuracy(self,testing_inputs,testing_outputs):     #will return the accuracy in percentage
        output_nodes = self.think(testing_inputs)   #generates the outputs using the testing inputs

        output = np.argmax(output_nodes, axis=1)    #one-hot-decodes the outputs to get the value
        correct = np.sum(output == testing_outputs)     #finds the amount of times the output matrix agrees with the expected output matrix
        accuracy = correct / np.size(testing_outputs) * 100     #returns fraction of correct values over total values
        return accuracy



if __name__ == "__main__":
    neural_network = NeuralNetwork()    #initialize the neural net



    inputs, outputs = mnist["data"], mnist["target"]

    outputs = outputs.astype(float)
    inputs = inputs / 255   #normalizing the grayscale to be between 0 and 1

    training_inputs,testing_inputs, training_outputs, testing_outputs = train_test_split(inputs, outputs,test_size = 0.15)  #split the data


    neural_network.train(training_inputs, training_outputs, 5000)   #do the training 5000 times, anything longer takes a lot of time
    print("Model accuracy: ")
    print(neural_network.test_accuracy(testing_inputs,testing_outputs))     #print the model accuracy

    #takes a while to run but I found that the accuracy is on average 90%
    #train over more iterations to acheive better accuracy
    #next step if I continue to work on this will be to implement one or two hidden layers


