'''
qmind_lib is a machine learning library written to assist us in hand-tracking needs.

Features:
- RHD dataset image loading and preparsing.
- Annotation loading for data from the RHD dataset.
'''

from venv import create
import numpy as np
import os
import time
import pickle
import sys 
import tensorflow as tf
import re
import matplotlib.pyplot as plt 

# NOTE: We note that the numbers 41258 and 2728 were retrieved directly from
# https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html
TRAIN_TOTAL_COUNT = 41258
EVALUATION_TOTAL_COUNT = 2728

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

def set_color_normal():
  print(bcolors.ENDC, end="")

def set_color_cyan():
  print(bcolors.OKCYAN, end="")

# converts a string into a cyan colored string
def cstr(str): 
  return bcolors.OKCYAN + str + bcolors.ENDC

# converts a string into a red colored string
def rstr(str): 
  return bcolors.FAIL + str + bcolors.ENDC

def get_anno(x, yt, ye):
  file_path = bytes.decode(x.numpy(), 'utf-8')
  re_match = re.search(r"\d{5}", file_path)
  img_index = int(file_path[re_match.start():re_match.end()])
  anno = yt[img_index]
  if "evaluation" in file_path:
    anno = ye[img_index]
  return anno

def download_label3D(file_path):
  # TODO(Noah): There must be better way than this!!!
  return tf.py_function(func=get_anno, inp=[file_path, y_train, y_test], Tout=tf.float32)

def download_label2D(file_path):
  # TODO(Noah): There must be better way than this!!!
  return tf.py_function(func=get_anno, inp=[file_path, y2_train, y2_test], Tout=tf.float32)

# Will "download" an image from the RHD dataset for us in tf.map
def download_image(file_path):

  global y2_test
  global y2_train  
  global rhd_root_dir

  # TODO(Noah): Do something about this here.
  GRAYSCALE=False
  IMAGE_SIZE=224

  img = tf.io.read_file(file_path)
  img = tf.io.decode_png(img, channels=3, dtype=tf.uint8)

  annot_2D = download_label2D(file_path)

  dC = annot_2D[0] - tf.constant([160,160], dtype=tf.float32) # how to get to the new image center.
  x_shift = tf.cast(dC[0], dtype=tf.int32)
  # Here we assume that downwards in pixel space gives a positive delta.
  y_shift = tf.cast(dC[1], dtype=tf.int32) 

  y_shift_prime = tf.math.maximum(y_shift, 0)
  x_shift_prime = tf.math.maximum(x_shift, 0)
  target_width = 320 - tf.math.abs(x_shift)
  target_height = 320 - tf.math.abs(y_shift)

  # works if both are positive.
  img = tf.image.crop_to_bounding_box(img, y_shift_prime, x_shift_prime, target_height, target_width)

  y_shift_prime = tf.math.maximum(-y_shift, 0)
  x_shift_prime = tf.math.maximum(-x_shift, 0)
  img = tf.image.pad_to_bounding_box(img, y_shift_prime, x_shift_prime, 320, 320)
  img = tf.image.resize(img, size=[IMAGE_SIZE, IMAGE_SIZE])

  return tf.cast(img, tf.float32)

def process_path(file_path):
  label = download_label3D(file_path)
  img = download_image(file_path)
  return img, label

# Used for both the training and the evaluation set.
def create_tf_dataset(img_dir, batch_size, img_count=-1):

  def configure_for_performance(ds):
    ds = ds.cache()
    # this is quite a big buffer! Presumably this gives us perfect shuffling :)
    ds = ds.shuffle(TRAIN_TOTAL_COUNT, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds 
  
  total_count = len(os.listdir(img_dir))
  if img_count == -1:
    img_count = total_count

  new_ds = tf.data.Dataset.list_files(
    os.path.join(img_dir, "*.png"), shuffle=False)
  skip_count = total_count - min(img_count, total_count)
  new_ds = new_ds.skip(skip_count)
  new_ds = new_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
  new_ds = configure_for_performance(new_ds)
  return new_ds

# this function must be called first thing when using qmind_lib
def init(rhd_dir, batch_size, img_count=-1):
  global rhd_root_dir
  global train_ds
  global eval_ds
  rhd_root_dir = rhd_dir

  np.set_printoptions(threshold=sys.maxsize)
  anno_train_path = os.path.join(rhd_dir, "anno", "anno_training.pickle")
  anno_eval_path = os.path.join(rhd_dir, "anno", "anno_evaluation.pickle")
  load_anno_all(anno_train_path, anno_eval_path)

  img_dir = os.path.join(rhd_root_dir, "training", "color")
  e_img_dir = os.path.join(rhd_root_dir, "evaluation", "color")
  train_ds = create_tf_dataset(img_dir, batch_size, img_count)
  # -1 for img_count loads ALL the data.
  eval_ds = create_tf_dataset(e_img_dir, batch_size, -1)

def visualize_ds():
  # gonna render for us a single batch! 
  image_batch, label_batch = next(iter(train_ds))
  plt.figure(figsize=(10, 10))
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.axis("off")
  plt.show()

def get_train_list():
  global rhd_root_dir
  return os.listdir(os.path.join(rhd_root_dir, "training", "color"))

def get_eval_list():
  global rhd_root_dir
  return os.listdir(os.path.join(rhd_root_dir, "evaluation", "color"))

y_train = np.zeros((TRAIN_TOTAL_COUNT, 21, 3), dtype=np.float32)
y_test = np.zeros((EVALUATION_TOTAL_COUNT, 21, 3), dtype=np.float32)
k_train = np.zeros((TRAIN_TOTAL_COUNT, 3, 3), dtype=np.float32)
k_test = np.zeros((EVALUATION_TOTAL_COUNT, 3, 3), dtype=np.float32)
y2_train = np.zeros((TRAIN_TOTAL_COUNT, 21, 2), dtype=np.float32)
y2_test = np.zeros((EVALUATION_TOTAL_COUNT, 21, 2), dtype=np.float32)

rhd_root_dir = None
train_ds = None
eval_ds = None

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

import imageio
import cv2
# Resizes an input image (numpy array) to another size.
def resize(img, size):
  return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

def download_image_legacy(set, img_index, GRAYSCALE=False, IMAGE_SIZE=224):

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