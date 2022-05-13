import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from mobilehand import camera_extrinsic, distance
from qmind_lib import get_train_list

# x and y should be standard lists!
def get_auc(x, y):
    result = 0
    for i in range(len(x)-1):
        delta_x = (x[i+1] - x[i]) / 30
        delta_y = y[i + 1] - y[i]
        result += 0.5 * delta_x * delta_y # add triangle area.
        result += delta_x * y[i] # rectangle @ bottom.
    return round(result, 3)    


# evaluates the model on a specific set from the RHD dataset.
def evaluate_model(model, tf_dataset):
    thresholds = [
        tf.constant([[0.02]]), tf.constant([[0.025]]), tf.constant([[0.03]]),
        tf.constant([[0.035]]), tf.constant([[0.04]]), tf.constant([[0.045]]),
        tf.constant([[0.05]])
    ]
    for i in range(len(thresholds)):
        thresholds[i] = tf.repeat(thresholds[i], repeats=21, axis=1)
    i = 0
    percentage = np.zeros((len(thresholds),))
    timings = []
    for threshold in thresholds:
        count = 0        
        total_img = 0        
        for img, label in tf_dataset:
            time_start = time.time()
            beta, pose, mesh, keypoints, scale = model(img)
            time_end = time.time()
            timings.append(time_end - time_start)
            keypoints = camera_extrinsic(scale, keypoints)
            error = distance(keypoints, label)
            bs = error.shape[0]
            _threshold = tf.repeat(threshold, repeats=bs, axis=0)
            valid_count = tf.math.count_nonzero(tf.math.less_equal(error, _threshold))
            count += valid_count.numpy()
            total_img += bs
        
        percentage[i] = count / (total_img * 21)
        i += 1
    
    thresholds = tf.stack(thresholds)
    thresholds = thresholds[:, 0] * 1000
    
    # Compute the average model inference time
    inference_time = np.sum(np.array(timings)) / (len(timings))
    print("average inference_time for one forward pass = {} s".format(round(inference_time, 3)))
    print("theoretical FPS = {}".format(round(1 / inference_time, 3)))

    plt.figure(figsize=(10.0, 8.0))
    plt.grid()

    _thresholds = [20, 25, 30, 35, 40, 45, 50]
    plt.plot(_thresholds, percentage, marker='x', label='Ours (AUC={})'.format(
        get_auc(_thresholds, percentage)
    ))
    
    __thresholds = [20, 23.33, 26.66, 30, 33.33, 36.66, 40, 43.33, 46.66, 50]
    vals = [0.75, 0.8, 0.84, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97]
    plt.plot(__thresholds, vals, linestyle=':', label = 'Zhang et al. (AUC={})'.format(
        get_auc(__thresholds, vals)
    ))
    vals =  [0.79, 0.87, 0.92, 0.95, 0.97, 0.98, 0.99]
    plt.plot(_thresholds, vals, marker='d', label = 'Baek et al. (AUC={})'.format(
        get_auc(_thresholds, vals)
    ))
    vals = [0.805, 0.87, 0.91, 0.93, 0.95, 0.955, 0.97] 
    plt.plot(_thresholds, vals, marker='s', label = 'Ge et al. (AUC={})'.format(
        get_auc(_thresholds, vals)
    ))

    plt.title('RHD Dataset 3D PCK Curve')
    plt.xlabel('Error Thresholds (mm)')
    plt.ylabel('Percentage of Correct Keypoints')
    plt.legend(loc="lower right")
    plt.show()
