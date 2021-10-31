# imports
import numpy as np # needed for matrix maths.
from sklearn.datasets import fetch_openml # function to fetch data from openml, open website hosting datasets
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import time
# imports

# fetch the training data 
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
print("before converting via to_categorical", y)
y = to_categorical(y)
print("after converting via to_categorical", y)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
# fetch the training data

# functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def relu(x):
    return np.maximum(0, x)

def init_weights_and_biases(n_inputs, n_outputs):
    return (0.01 * np.random.randn(n_inputs, n_outputs), np.zeros(1, n_outputs) )

def dense_layer_forward(w, b, a):
    return np.dot(a, w) + b
# functions

# build network
w1, b1 = init_weights_and_biaes(784, 16)
w2, b2 = init_weights_and_biaes(16, 16)
w3, b3 = init_weights_and_biaes(16, 10)

# set some parameters
learning_rate = 0.001
epochs = 10

# train the network
start_time = time.time()
for n in range(1, epochs + 1):
    print('Starting epoch', n)
    epoch_start_time = time.time()
    for x,y in zip(x_train, y_train): # do not even think that there is bactching going on here. Just think that we are training on the entire set.
        #forward pass of network
        y_pred1 = dense_layer_forward(w1, b1, x)
        y_pred2 = sigmoid(y_pred1)
        y_pred3 = dense_layer_forward(w2, b2, y_pred2)
        y_pred4 = sigmoid(y_pred3)
        y_pred5 = dense_layer_forward(w3, b3, y_pred4)
        y_pred6 = sigmoid(y_pred5)
        # calculate the gradients (note we are not training the bias and thus they will always remain zero)
        grad = {}
        error = 2 * ( y - y_pred6 ) / y_pred6.shape[0] # mean squared error loss contribution
        error *= sigmoid_derivative(y_pred5) # contribution from sigmoid derivative
        grad['w3'] = np.outer(error, y_pred4)
        error = np.dot(w3.T, error) * sigmoid_derivative(y_pred3)
        grad['w2'] = np.outer(error, y_pred2])
        error = np.dot(w2.T, error) * sigmoid_derivative(y_pred1)
        grad['w1'] = np.outer(error, z)
        # update the network parameters
        for key, value in grad.items():
            arr = w3    
            if (key != 'w3'):
                if (key == 'w2'):
                    arr = w2
                elif (key == 'w1'):
                    arr = w1
            arr -= learning_rate * value # gradient descent. No momentum. Nothing fancy here.
    epoch_end_time = time.time()
    print('Epoch took', epoch_end_time - epoch_start_time, "many seconds.")
    # compute the accuracy of the model now for testing purposes.
    predictions = []
    for x, y in zip(x_val, y_val):
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

end_time = time.time()
print('Total training took', end_time - start_time, "many seconds.")