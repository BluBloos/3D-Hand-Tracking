# The training set examples are as follows.
# -> in the "training/color" folder, we have 00000.png through until 41257 png. Each again are 320x320 pixels.
# -> in the "training/mask" folder, we have 00000.png through until 41257 png. Each again are 320x320 pixels.
# -> in the "evaluation/color" folder, we have 00000.png through until 02727.png. Each again are 320z320 pixels. Suppposed to be testing data.
# -> in the "evaluation/mask" folder, we have 00000.png through until 02727.png. Each again are 320x320 pixels. Supposed to be testing data.

# import libraries
import os
import pickle
import matplotlib.pyplot as plt
import imageio
import numpy as np
import time
import math
#import libraries

dir = 'RHD_published_v2'
set = 'training'
path = os.path.join(dir, set)
with open(os.path.join(path, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# EXAMPLE CODE FOR LOADING IN THE ANNOTATIONS OF THE DATASET
for sample_id, anno in anno_all.items():
    image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
     
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel.
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean

    # EXAMPLE ON HOW TO VISUALIZE DATA
    # plt.imshow(image)
    # plt.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    # plt.show()

    break 
# EXAMPLE CODE FOR LOADING IN THE ANNOTATIONS OF THE DATASET

# EXAMPLE CODE FOR CONVERTING A SEGMENTATION MASK TO THE PROPER FORMAT
for sample_id, anno in anno_all.items():
    # For the formatting of the mask data...
    # 0: background, 1: person, 
    # 2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
    # 18-20: right thumb, ..., right palm: 33
    mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))
    # Convert to the mask that we desire! (0 for anything that is not a hand, and 1 for anything that is a hand)
    mask[mask == 1] = 0
    mask[mask >= 2] = 1
    break 
# EXAMPLE CODE FOR CONVERTING A SEGMENTATION MASK TO THE PROPER FORMAT

# BELOW IS A QUICK AND DIRTY ALGO TO RELIABLY REJECT ALL TRAINING AND TESTING SAMPLES WHERE THERE ARE TWO HANDS IN THE IMAGE
# How do we do this?
# -> Look at the kp_visible array. Pick examples from the training set where only points 
# from the left hand can be seen or only points from the right hand
# can be seen. It does not need to be the case where all points for 
# just one hand are seen. It can be a subset of points, but so long as that subset belongs
# to just one hand.
total_training_examples = 41257 + 1
valid_training_examples = 0
start_time = time.time()
print("Begin single hand parse")
for sample_id, anno in anno_all.items():
    ''' format of the kp_visible array
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    '''
    kp_visible = (anno['uv_vis'][:, 2] == 1)
    case1 = np.sum(kp_visible[0:21])
    case2 = np.sum(kp_visible[21:])
    # also invalidates training examples where none of the hands can be seen. We must see at least some of just one hand for it to be valid.
    valid_case = (case1 > 0 and case2 == 0) or (case1 == 0 and case2 > 0) 
    if (valid_case):
        valid_training_examples += 1
end_time = time.time()
print("Total elapsed time for single hand parse =", end_time - start_time, "s")
print("Amount of valid training examples = ", valid_training_examples / total_training_examples * 100, "%")
# END OF QUICK AND DIRTY ALGO TO REJECT ALL TWO HAND DATA POINTS

# This function will take in a uv coordinate and generate the corresponding heatmap!!
def HeatmapFromUV(uv):
    # how exactly do we generate a gaussian function again??
    # f(x) = a * e ^ ( -(x-b)^2 / 2c^2 )
    # a is the height of the curves peak (for us this will be the pixel intensity)
    # b is the pos of the center of the peak (this will be the uv coordinate)
    # c is the standard deviation (controls the width of the gaussian = 4 px)
    heatmap = [] 
    
    # gaussian parameters
    # According to the paper: https://arxiv.org/pdf/1903.00812.pdf
    # The ground truth heat-map is defined as a 2D Gaussian with
    # a standard deviation of 4 px centered on the ground truth 2D
    # joint location
    a = 255
    c = 4
    b_x = uv[0]
    b_y = uv[1] 

    # thinking a 3D gaussian is just like a radial type ting. Like you rotate the 2D guassian and that's how you 
    # get the 3D one :)
    # we love hardcoding things :)
    for x in range(320):
        new_row = []
        for y in range(320):
            R = (x - b_x)**2 + (y - b_y)**2
            val = a * math.exp( -(R) / 2 / c**2 ) 
            new_row.append( val )
        heatmap.append(new_row)    
    
    np_heatmap = np.array(heatmap)
    return np_heatmap

# Stole this function from Stackoverflow :)
def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)

# ALL CODE BELOW IS NOW THE MODEL IMPLEMENTATION + TRAINING CODE
#NOTE: Following along with -> https://www.tensorflow.org/tutorials/quickstart/advanced
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras import Model

model = tf.keras.models.Sequential([
  Conv2D(32, 8, activation='relu', input_shape=(320,320, 1), data_format="channels_last", padding="same"),
  MaxPool2D(),  
  Conv2D(16, 4, activation = 'relu', padding="same"),
  MaxPool2D(),  
  Conv2D(1, 2, activation = 'sigmoid', padding="same"), 
  UpSampling2D(),
  UpSampling2D()
])

# LOAD IN THE FIRST IMAGE IN THE DATASET FOR TESTING (AS WELL AS FOR REDRAWING PREDs AFTER EACH EPOCH)
image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % 0)) 
_image = image.astype('float32')
_image = rgb2gray(_image / 255)
_image = np.expand_dims(_image, axis=0 )
print("_image", _image)
print("_image.shape", _image.shape)

# LOAD IN THE DATA
x_train = np.zeros( (1000, 320, 320, 1) ) # 1 channel on the input for grayscale images!
y_train = np.zeros( (1000, 320, 320, 1) )
x_test = np.zeros( (100, 320, 320, 1) ) # 1 channel on the input for grayscale images!
y_test = np.zeros( (100, 320, 320, 1) ) 

def LoadData(dataAmount, dataType, np1, np2):
    path = os.path.join(dir, dataType)
    with open(os.path.join(path, 'anno_%s.pickle' % dataType), 'rb') as fi:
        anno_all = pickle.load(fi)

    count = 0 # count is the amount of data items that have been loaded thus far.
    for sample_id, anno in anno_all.items():
        image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
        _image = image.astype('float32')
        _image = rgb2gray(_image / 255)
        np1[count, :, :, :] = _image
        mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))
        # augment the mask
        mask[mask == 1] = 0
        mask[mask >= 2] = 255
        _mask = mask.astype('float32')
        _mask = _mask / 255
        np2[count, :, :, :] = np.expand_dims(_mask, axis=2)

        count += 1
        if (count >= dataAmount):
            break

print("Loading in the training data samples...")
start_time = time.time()
LoadData(1000, 'training', x_train, y_train)
end_time = time.time()
print('Elapsed for LoadData training', end_time - start_time, 's')
print("Loading in the evaluation data samples...")
start_time = time.time()
LoadData(100, 'evaluation', x_test, y_test)
end_time = time.time()
print('Elapsed for LoadData evaluation', end_time - start_time, 's')

# Batch the data for tensorflow.
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model.summary() # print out the model in a nice, clean format.

class StupidSimpleLossMetric():
    def __init__(self):
        self.losses = [] # empty python array 
    def __call__(self, loss):
        self.losses.append(loss)
    def result(self):
        return sum(self.losses) / len(self.losses)
    def reset_states(self):
        self.losses = []


# KLDivergence gets us the difference between two probability distributions
#loss_object = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# Black box -> takes the weights and gradients and gives us better weights :)
optimizer = tf.keras.optimizers.Adam() # defaults should work just fine

train_loss = StupidSimpleLossMetric()
test_loss = StupidSimpleLossMetric()

train_accuracy = tf.keras.metrics.MeanAbsolutePercentageError(name='train_accuracy')
test_accuracy = tf.keras.metrics.MeanAbsolutePercentageError(name='test_accuracy')

# @tf.function Compiles a function into a callable TensorFlow graph.
# https://www.tensorflow.org/guide/intro_to_graphs
def loss_func(pred, labels):
    return tf.math.reduce_sum( tf.math.square(labels - tf.cast(pred,tf.float64)))

#@tf.function
def train_step(images, segmentation_masks):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_func(predictions, segmentation_masks)
        #loss = np.dot(tf.reshape(segmentation_masks, [102400], tf.reshape(predictions, [102400])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# test the model after training it
#@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_func(predictions, labels)
  test_loss(t_loss)
  test_accuracy(labels, predictions) 


EPOCHS = 10 # sure...

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  print("Epoch", epoch)
  start = time.time()
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  end = time.time()

  print(
    f'Epoch {epoch + 1}, '
    f'Time {end-start} s'
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

  # for each epoch, we want to show the 
  pred = model( _image )
  plt.imshow(pred[0])
  plt.show()