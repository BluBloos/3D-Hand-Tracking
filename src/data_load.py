########################################### DATA LOADING ###########################################
IMAGE_SIZE = 224
GRAYSCALE = False
IMAGE_CHANNELS = 1 if GRAYSCALE else 3

import numpy as np
import cv2
import os
import imageio

from anno_load import y2_test
from anno_load import y2_train

# NOTE(Noah): Stole this function from Stackoverflow :)
def rgb2gray(rgb):
  return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)
    
def resize(img, size):
  return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
  
def download_image(root_dir, set, img_index):

  global y2_test
  global y2_train  

  file_path = os.path.join(root_dir, set, "color", "{:05d}.png".format(img_index))
  image = imageio.imread(file_path)
  _image = image.astype('float32')
  if GRAYSCALE:
    _image = rgb2gray(_image)
  else:
    _image = _image

  annot_2D = y2_train[img_index]
  if set == "evaluation":
    annot_2D = y2_test[img_index]
  pixel_trans = np.array([160,160]) - annot_2D[0]
  x_shift = int(pixel_trans[0])
  y_shift = int(pixel_trans[1])
  _image = np.roll( _image, (y_shift, x_shift), axis=(0,1) )

  # black out the regions we do not care about
  if y_shift > 0:
    _image = cv2.rectangle(_image, (0, 0), (320, y_shift), 0, -1)
  else:
    _image = cv2.rectangle(_image, (0, 320 + y_shift), (320, 320), 0, -1)

  if x_shift > 0:
    _image = cv2.rectangle(_image, (0, 0), (x_shift, 320), 0, -1)
  else:
    _image = cv2.rectangle(_image, (320 + x_shift, 0), (320, 320), 0, -1)

  _image = resize(_image, IMAGE_SIZE)

  return _image
########################################### DATA LOADING ###########################################