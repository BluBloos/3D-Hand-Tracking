# Need to read in the RGB images and segmentation masks
# Need to mode the segmentation masks to be just black and white
# Need to write a quick "find the amount of islands" algo to reject all training and testing examples where there are two hands in the photo

# Have an intermediate loss term for learning the segmentaton masks.
# Then we have a final loss term for the full 21 joints prediction.

# Thinking right of the bat that we convert the ground truth keypoints to gaussian heatmaps. This seems like it would be easier to learn...
# -> gives the network more room to play around!

# The training set examples are as follows.
# -> in the "training/color" folder, we have 00000.png through until 41257 png. Each again are 320x320 pixels.
# -> in the "training/mask" folder, we have 00000.png through until 41257 png. Each again are 320x320 pixels.
# -> in the "evaluation/color" folder, we have 00000.png through until 02727.png. Each again are 320z320 pixels. Suppposed to be testing data.
# -> in the "evaluation/mask" folder, we have 00000.png through until 02727.png. Each again are 320x320 pixels. Supposed to be testing data.

# What about the ground truth data?

# in /training/anno_training.pickle, this is a Python data structure containing the keypoint annotations and camera matrices K.
# cool, but what is the format and how do I read it?

# Load annotations of this set
import os
import pickle
import matplotlib.pyplot as plt
import imageio
import numpy as np
import time
import math

dir = 'RHD_published_v2'
set = 'training'
path = os.path.join(dir, set)
with open(os.path.join(path, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

for sample_id, anno in anno_all.items():
    image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
    print("Image", image)
    
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel.
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
    print("kp_coord_uv", kp_coord_uv)
    print("kp_visible", kp_visible)

    # Visualize data
    #plt.imshow(image)
    #plt.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    #plt.show()

    break # Close program after just one sample (we are unit testing after all)

# UNIT TEST #1.
# Read in just one training example and plot the example data with the 2D keypoints right on top.
# DONE.

# Next step -> convert the segmentation masks to JUST black and white.
# UNIT TEST #2

for sample_id, anno in anno_all.items():
    
    # For the formatting of the mask data...
    # 0: background, 1: person, 
    # 2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
    # 18-20: right thumb, ..., right palm: 33
    mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))
    print("mask", mask)
    print("mask.shape", mask.shape)

    #plt.imshow(mask)
    #plt.show()

    # Convert to the mask that we desire! (0 for anything that is not a hand, and 1 for anything that is a hand)
    mask[mask == 1] = 0
    mask[mask >= 2] = 1

    #plt.imshow(mask)
    #plt.show()

    break # just need one test, real simple.

# NEXT, next step, -> write a quick and dirty algorithm to reliably reject all training and testing samples where there are two hands in the image...
# How do we do this?
# -> Look at the kp_visible array. Pick only example in the training set where only points from the left hand can be seen or only points from the right hand
# can be seen. It does not need to be the case where all points for just one hand are seen. It can be a subset of points, but so long as that subset belongs
# to just one hand.

total_training_examples = 41257 + 1
valid_training_examples = 0
start_time = time.time()

print("Begin single hand parse")
for sample_id, anno in anno_all.items():
    # format of the kp_visible array
    '''
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    '''
    kp_visible = (anno['uv_vis'][:, 2] == 1)
    case1 = np.sum(kp_visible[0:21])
    case2 = np.sum(kp_visible[21:])
    #print("kp_visible", kp_visible)
    #print("case1", case1)
    #print("case2", case2)

    # also invalidates training examples where none of the hands can be seen. We must see at least some of just one hand for it to be valid.
    valid_case = (case1 > 0 and case2 == 0) or (case1 == 0 and case2 > 0) 
    if (valid_case):
        valid_training_examples += 1



end_time = time.time()
print("Total elapsed time for single hand parse =", end_time - start_time, "s")
print("Amount of valid training examples = ", valid_training_examples / total_training_examples * 100, "%")

# Another unit test, going to need to generate a heatmap for the estimation of a keypoint!!
# What is the format of the heatmap that we care about?

# According to the paper: https://arxiv.org/pdf/1903.00812.pdf
# The ground truth heat-map is defined as a 2D Gaussian with
# a standard deviation of 4 px centered on the ground truth 2D
# joint location

# how exactly do we generate a gaussian function again??
# f(x) = a * e ^ ( -(x-b)^2 / 2c^2 )
# a is the height of the curves peak (for us this will be the pixel intensity)
# b is the pos of the center of the peak (this will be the uv coordinate)
# c is the standard deviation (controls the width of the gaussian = 4 px)

# This function will take in a uv coordinate and generate the corresponding heatmap!!
def HeatmapFromUV(uv):
    heatmap = [] 
    
    # gaussian parameters
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
    #plt.imshow(np_heatmap)
    #plt.show()

HeatmapFromUV(np.array([ 100, 100 ]))

# Further yet testing of the heatmap from UV function!
for sample_id, anno in anno_all.items():
    #image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
    #print("Image", image)
    
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel.
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
    print("kp_coord_uv", kp_coord_uv)
    print("kp_visible", kp_visible)

    # Visualize data
    # plt.imshow(image)
    # plt.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    # plt.show()

    # go for each visible uv coord and plot that shit!!!
    '''for i in range( kp_coord_uv.shape[0] ):
        coord = kp_coord_uv[i]
        visible = kp_visible[i]
        if (visible):
            HeatmapFromUV(coord)
    '''

    break # Close program after just one sample (we are unit testing after all)


# Model architecture brainstorm.
# 320x320 images. RGB (color) -> 3 cannels.
# Run a fully connected conv net right on this.



# OKAY, ... NOW we try our hands at building this model...


#NOTE: Following along with -> https://www.tensorflow.org/tutorials/quickstart/advanced


import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras import Model


'''
#NOTE: Make the training and test datasets...
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
'''

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # 32 filters, 3x3 kernel
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        self.conv1 = Conv2D(32, 8, activation='relu', input_shape=(1, 320,320,3), padding="same")
        self.conv2 = Conv2D(16, 4, activation = 'relu', padding="same")
        self.conv3 = Conv2D(1, 2, activation = 'sigmoid', padding="same")
        self.upsample = UpSampling2D() # Default factor of 2x
        self.maxpooling = MaxPool2D() # default of (2, 2) pool and strid of (2, 2)
        #self.conv1 = Conv2D(32, 3, activation='relu')
        #self.flatten = Flatten()
        #self.d1 = Dense(128, activation='relu')
        #self.d2 = Dense(10)

    def call(self, x):

        # Takes an input x, 320x320 pixel array.
        # Want 3 convolution layers with, 8, 4, 2 kernel sizes for convolution
        # Stride = 1 
        # Then size 2 for all max pooling layers.

        x = self.conv1(x) # activation is included with this conv.
        #print("after conv1: tf(x.shape())", tf.shape(x))
        x = self.maxpooling(x)
        #print("after maxpooling: tf(x.shape())", tf.shape(x))
        x = self.conv2(x)
        #print("after conv2: tf(x.shape())", tf.shape(x))
        x = self.maxpooling(x)
        #print("after maxpooling: tf(x.shape())", tf.shape(x))
        x = self.conv3(x)
        #print("after conv3: tf(x.shape())", tf.shape(x))
        x = self.upsample(x) # No parameters here
        x = self.upsample(x)
        #print("after upsample: tf(x.shape())", tf.shape(x))
        #x = self.conv1(x)
        #x = self.flatten(x)
        #x = self.d1(x)
        
        return x


# create the model instance
model = MyModel()

image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % 0)) # load in the first image from the dataset
_image = np.expand_dims(image, axis=0 )
print("_image", _image)
print("_image.shape", _image.shape)
_image = _image.astype(float)
_image = _image / 255

pred = model( _image )

print( tf.shape(pred) )
print(pred.numpy())

#plt.imshow(pred.numpy()[0])
#plt.show()


# train_ds and test_ds

# x_train.shape = (5000, 320, 320, 3)
x_train = np.zeros( (200, 320, 320, 3) )
y_train = np.zeros( (200, 320, 320, 1) )
x_test = np.zeros( (100, 320, 320, 3) )
y_test = np.zeros( (100, 320, 320, 1) ) 

# load in the data
def LoadData(dataAmount, dataType, np1, np2):
    path = os.path.join(dir, dataType)
    with open(os.path.join(path, 'anno_%s.pickle' % dataType), 'rb') as fi:
        anno_all = pickle.load(fi)

    count = 0
    for sample_id, anno in anno_all.items():
        image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
        _image = image.astype(float)
        _image = _image / 255
        np1[count, :, :, :] = _image
        mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))
        # augment the mask
        mask[mask == 1] = 0
        mask[mask >= 2] = 255
        _mask = mask.astype(float)
        _mask = _mask / 255
        np2[count, :, :, :] = np.expand_dims(_mask, axis=2)

        count += 1
        if (count >= dataAmount):
            break

start_time = time.time()
LoadData(200, 'training', x_train, y_train)
end_time = time.time()
print('Elapsed for LoadData training', end_time - start_time, 's')
start_time = time.time()
LoadData(100, 'evaluation', x_test, y_test)
end_time = time.time()
print('Elapsed for LoadData evaluation', end_time - start_time, 's')

#plt.imshow(x_train[0])
#plt.show()
#plt.imshow(y_train[0])
#plt.show()

#print('x_train', x_train.shape)
#print('y_train', y_train.shape)

    


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

'''
'''


# TRAINING MODEL CODE

# KLDivergence gets us the difference between two probability distributions
loss_object = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# Black box -> takes the weights and gradients and gives us better weights :)
optimizer = tf.keras.optimizers.Adam() # defaults should work just fine

train_loss = tf.keras.metrics.KLDivergence(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.KLDivergence(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, segmentation_masks):
    with tf.GradientTape() as tape:
        predictions = model(images)
        #loss = np.dot(tf.reshape(segmentation_masks, [102400], tf.reshape(predictions, [102400])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss, predictions)
    train_accuracy(labels, predictions)

    #print("train_step loss:", train_loss.result())
    #print("train_step accuracy:", train_accuracy.result() * 100)

# test the model after training it
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss, predictions)
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

  pred = model( _image )
  plt.imshow(pred[0])
  plt.show()