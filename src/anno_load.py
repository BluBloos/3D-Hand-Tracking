########################################### LOAD ANNOTATIONS ###########################################
import os
import numpy as np
import pickle
import time
from qmindcolors import cstr

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
    # i.e. we want to densely store the annotations.
    if(not leftHand):
        y[count, :, :] = np.array(value['xyz'][21:42], dtype=np.float32)
        y2[count, :, :] = np.array(value['uv_vis'][:, :2][21:42], dtype=np.float32)
    else: 
        y[count, :, :] = np.array(value['xyz'][:21], dtype=np.float32)
        y2[count, :, :] = np.array(value['uv_vis'][:, :2][:21], dtype=np.float32)

    # Adjust the 3D keypoints to be at the center of the image.
    annot_3D = y[count, :, :]
    y[count, :, :] -= np.array([annot_3D[0][0], annot_3D[0][1], 0.0], dtype=np.float32)

    k[count, :, :] = value['K']
    count += 1
########################################### LOAD ANNOTATIONS ###########################################