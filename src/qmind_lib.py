'''
qmind_lib is a machine learning library written to assist us in hand-tracking needs.

Features:
- RHD dataset image loading and preparsing.
- Annotation loading for data from the RHD dataset.
'''

import numpy as np
import cv2
import os
import imageio
import time
import pickle
import sys 

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

# converts a string into a cyan colored string
def cstr(str): 
  return bcolors.OKCYAN + str + bcolors.ENDC

# converts a string into a red colored string
def rstr(str): 
  return bcolors.FAIL + str + bcolors.ENDC

# this function must be called first thing when using qmind_lib
def init(rhd_dir, debug_mode=True):
  global rhd_root_dir
  rhd_root_dir = rhd_dir
  np.set_printoptions(threshold=sys.maxsize)
  anno_train_path = os.path.join(rhd_dir, "anno", "anno_training.pickle")
  anno_eval_path = os.path.join(rhd_dir, "anno", "anno_evaluation.pickle")
  load_anno_all(anno_train_path, anno_eval_path)

def get_train_list():
  global rhd_root_dir
  return os.listdir(os.path.join(rhd_root_dir, "training", "color"))

def get_eval_list():
  global rhd_root_dir
  return os.listdir(os.path.join(rhd_root_dir, "evaluation", "color"))

# NOTE: We note that the numbers 41258 and 2728 were retrieved directly from
# https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html
TRAIN_TOTAL_COUNT = 41258
EVALUATION_TOTAL_COUNT = 2728

y_train = np.zeros((TRAIN_TOTAL_COUNT, 21, 3), dtype=np.float32)
y_test = np.zeros((EVALUATION_TOTAL_COUNT, 21, 3), dtype=np.float32)
k_train = np.zeros((TRAIN_TOTAL_COUNT, 3, 3), dtype=np.float32)
k_test = np.zeros((EVALUATION_TOTAL_COUNT, 3, 3), dtype=np.float32)
y2_train = np.zeros((TRAIN_TOTAL_COUNT, 21, 2), dtype=np.float32)
y2_test = np.zeros((EVALUATION_TOTAL_COUNT, 21, 2), dtype=np.float32)

rhd_root_dir = None

def load_anno_all(anno_train_path, anno_eval_path):
  global y_train
  global y_test
  global k_train
  global k_test
  global y2_train
  global y2_test
  print("Loading in training annotations")
  time_start = time.time()
  load_anno(anno_train_path, y_train, k_train, y2_train)
  time_end = time.time()
  print(cstr("Training annotations loaded in {} s".format(time_end - time_start)))
  print("Loading in evaluation annotations")
  time_start = time.time()
  load_anno(anno_eval_path, y_test, k_test, y2_test)
  time_end = time.time()
  print(cstr("Evaluation annotations loaded in {} s".format(time_end - time_start)))

def load_anno(path, y, k, y2):
  anno_all = []
  count = 0
  with open(path, 'rb') as f:
    anno_all = pickle.load(f)
  for key, value in anno_all.items():
    kp_visible = (value['uv_vis'][:, 2] == 1)
    case1 = np.sum(kp_visible[0:21])
    case2 = np.sum(kp_visible[21:])
    leftHand = case1 > 0
    # NOTE: We note here that we are not checking if this training or evaluation example is valid.
    # i.e. we want to store the annotations dense.
    if(not leftHand):
        y[count, :, :] = np.array(value['xyz'][21:42], dtype=np.float32)
        y2[count, :, :] = np.array(value['uv_vis'][:, :2][21:42], dtype=np.float32)
    else: 
        y[count, :, :] = np.array(value['xyz'][:21], dtype=np.float32)
        y2[count, :, :] = np.array(value['uv_vis'][:, :2][:21], dtype=np.float32)

    # Adjust the 3D keypoints to be at the center of the image.
    annot_3D = y[count, :, :]
    y[count, :, :] -= annot_3D[0]

    k[count, :, :] = value['K']
    count += 1

# Converts an input rgb image (numpy array) to a grayscale image.
def rgb2gray(rgb):
  return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)

# Resizes an input image (numpy array) to another size.
def resize(img, size):
  return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

# Will "download" an image from the RHD dataset. set is one of either "training" or "evaluation". 
# if for example the filename of the image is 00018.png, img_index should be set to 18.
# 
# The caller of this function is agnostic to specific implementation. This function might download a file
# via the network, might load from cache, or might open a file from disk.
# 
# For appropriate operation, call init_RHD() first. 
def download_image(set, img_index, GRAYSCALE=False, IMAGE_SIZE=224):

  global y2_test
  global y2_train  
  global rhd_root_dir

  if (rhd_root_dir == None):
    raise Exception("Please call init_RDH() prior to calling download_image()")

  file_path = os.path.join(rhd_root_dir, set, "color", "{:05d}.png".format(img_index))
  image = imageio.imread(file_path)

  annot_2D = y2_train[img_index]
  if set == "evaluation":
    annot_2D = y2_test[img_index]
  pixel_trans = np.array([160,160]) - annot_2D[0]
  x_shift = int(pixel_trans[0])
  y_shift = int(pixel_trans[1])

  _image = np.roll( image, (y_shift, x_shift), axis=(0,1) )

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

  return _image.astype(np.float32)
