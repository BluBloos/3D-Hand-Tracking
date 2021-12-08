# Imports
import numpy as np # needed for matrix maths.
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml # function to fetch data from openml, open website hosting datasets#
from sklearn.model_selection import train_test_split
import time
# Imports

# Little bit of a unit test for numpy dot operation.
v = np.array(
    [3.1, 2.2, 4.3]
)

M = np.array( 
    [ 
        [ 2.5, 4.1, 5.4 ],
        [ 3.9, 1.01, 3.4 ],
        [ 4.3, 2.3, 1.6 ]
    ] 
)
print('v.shape', v.shape)
print('M.shape', M.shape)
print('Mv', np.dot(M, v ))
#END of numpy.dot unit test.

# Now we fetch the data from openml and augment the data
mnist = fetch_openml('mnist_784')
inputs, outputs = mnist["data"], mnist["target"]

print('inputs', inputs)
x = inputs.to_numpy()
print('x', x)

print('outputs', outputs)
y = outputs.to_numpy(dtype=float)
print('y', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15) # split the data
print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

def to_one_hot(inputVar):  # function that does one-hot-encoding on the expected output
    transformation_array = np.array([[1 / 11], [1], [1 / 2], [1 / 3], [1 / 4], [1 / 5], [1 / 6], [1 / 7], [1 / 8], [1 / 9]])
    inputVar = np.expand_dims(inputVar, axis=1)
    indicator = np.dot(inputVar, transformation_array.T)     # will leave a 1 in the correct spot for each entry
    indicator[np.all((indicator == 0), axis=1)] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]     #every all-zero row set to [1,0,0,0,0,0,0,0,0,0]
    indicator[indicator != 1] = 0   # everything not a 1 set to zero
    return indicator    #returns the one-hot-encoded expected output

y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

print('y_train_onehot', y_train)
print('y_test_onehot', y_test)
# DONE fetching and augmenting the data.

# functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def relu(x):
    return np.maximum(0, x)

def init_weights_and_biases(n_inputs, n_outputs):
    # np.zeros takes in a shape but np.random.randn does not!
    return ( 0.01 * np.random.randn(n_outputs, n_inputs), np.zeros( (n_outputs) ) )

def dense_layer_forward(w, b, a):
    return np.dot(w, a) + b
# functions

# build network
w1, b1 = init_weights_and_biases(784, 16)
w2, b2 = init_weights_and_biases(16, 16)
w3, b3 = init_weights_and_biases(16, 10)

print('w1', w1)
print('w1.shape', w1.shape)
print('x_train[0]', x_train[0])
print('b1', b1)

y_pred1 = dense_layer_forward(w1, b1, x_train[0])
print('y_pred1', y_pred1)

# set some parameters
learning_rate = 0.01
epochs = 10

# train the network
start_time = time.time()
for n in range(1, epochs + 1):
    print('Starting epoch', n)
    epoch_start_time = time.time()
    for x,y in zip(x_train, y_train): # do not even think that there is bactching going on here. Just think that we are training on the entire set.
        # forward pass of network
        y_pred1 = dense_layer_forward(w1, b1, x)
        y_pred2 = sigmoid(y_pred1)
        y_pred3 = dense_layer_forward(w2, b2, y_pred2)
        y_pred4 = sigmoid(y_pred3)
        y_pred5 = dense_layer_forward(w3, b3, y_pred4)
        y_pred6 = sigmoid(y_pred5)
        # calculate the gradients (note we are not training the bias and thus they will always remain zero)
        grad = {}
        error = 2 * ( y - y_pred6 ) # mean squared error loss contribution, batch size of 1.
        #print('error =  2 * (y - y_pred6)', error)
        error = np.multiply( error, sigmoid_derivative(y_pred5) ) # contribution from sigmoid derivative at the end.
        #print('error * sigmoid_dervivative', error)
        grad['w3'] = np.outer(error, y_pred4) # checks out!
        error = np.dot(w3.T, error) * sigmoid_derivative(y_pred3) # NOTE: HAVE NOT TAKEN A LOOK AT THIS CODE.
        grad['w2'] = np.outer(error, y_pred2)
        error = np.dot(w2.T, error) * sigmoid_derivative(y_pred1)
        grad['w1'] = np.outer(error, x)
    
        # print out the deltas
        #print('grad["w3"] * learning_rate', grad["w3"] * learning_rate)
        #print('grad["w2"] * learning_rate', grad["w2"] * learning_rate)
        #print('grad["w1"] * learning_rate', grad["w1"] * learning_rate)

        # update the network parameters
        # gradient descent, no momentum. Nothing fancy here.
        # NOTE: We know that this portion was unit tested, and it works.
        for key, value in grad.items():  
            if (key == 'w3'):
                w3 -= learning_rate * value
            elif (key == 'w2'):
                w2 -= learning_rate * value
            elif (key == 'w1'):
                w1 -= learning_rate * value

        #exit() # just quit the program because I want to see what happens here...


    epoch_end_time = time.time()
    print('Epoch took', epoch_end_time - epoch_start_time, "many seconds.")
    # compute the accuracy of the model now for testing purposes.
    predictions = []
    for x, y in zip(x_test, y_test):
        # TODO: Make a function for the forward pass of the network...
        y_pred1 = dense_layer_forward(w1, b1, x)
        y_pred2 = sigmoid(y_pred1)
        y_pred3 = dense_layer_forward(w2, b2, y_pred2)
        y_pred4 = sigmoid(y_pred3)
        y_pred5 = dense_layer_forward(w3, b3, y_pred4)
        y_pred6 = sigmoid(y_pred5)
        # TODO: Make a function for the forward pass of the network...
        pred = np.argmax(y_pred6) # find the prediction via the one-hot encoding.
        predictions.append(pred == np.argmax(y))
    print('Accuracy', np.mean(predictions))
    # pick a sample to plot
    sample = 1
    image = x_train[sample]# plot the sample
    fig = plt.figure
    plt.imshow(image.reshape((28, 28)), cmap='gray')
    plt.show()
    # also print the pred for this
    y_pred1 = dense_layer_forward(w1, b1, x) # transpose X to make it a vertical vector for matrix multiplication :)
    y_pred2 = sigmoid(y_pred1)
    y_pred3 = dense_layer_forward(w2, b2, y_pred2)
    y_pred4 = sigmoid(y_pred3)
    y_pred5 = dense_layer_forward(w3, b3, y_pred4)
    y_pred6 = sigmoid(y_pred5)
    print('y_pred6', y_pred6)

end_time = time.time()
print('Total training took', end_time - start_time, "many seconds.")