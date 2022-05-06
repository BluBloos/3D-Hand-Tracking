import numpy as np
import matplotlib.pyplot as plt
from mobilehand_lfuncs import LOSS_2D, LOSS_3D, LOSS_REG
import os
import tensorflow as tf
from mobilehand import camera_extrinsic
from mobilehand_lfuncs import distance


# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory that contains every single evaluation image.
# download_image is a function that we can call (we pass it one param: fileName), and 
#   it will return to us a numpy array for the image.
def time_model(model, rhd_eval_dir, download_image):
    pass


# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory name for a directory 
#   that contains every single evaluation image (of just single hands).
# download_image is a function that we can call.
# y_test is a numpy array with the 3D keypoints for every single image in the RHD evaluation set.
#   the indices into this array are the names of the image files.
def evaluate_model(model, rhd_eval_dir, set, download_image, y_test, gcs_path):
    length = len(os.listdir(rhd_eval_dir))
    thresholds = [
        tf.repeat(0.02,repeats = 21), tf.repeat(0.025,repeats = 21), tf.repeat(0.03,repeats = 21),
        tf.repeat(0.035,repeats = 21), tf.repeat(0.04,repeats = 21), tf.repeat(0.045,repeats = 21),
        tf.repeat(0.05,repeats = 21)
    ]
    i = 0
    percentage = np.zeros((len(thresholds),))
    
    for threshold in thresholds:

        count = 0        
        for filename in os.listdir(rhd_eval_dir):
            
            print(filename)
            index = int(filename[0:5])
            annot_3D = y_test[index]
            image = download_image(gcs_path, set, index)
            image = tf.expand_dims(image, axis = 0)

            beta, pose, mesh, keypoints, scale = model(image)
            keypoints = camera_extrinsic(scale, keypoints)
            error = distance(keypoints, annot_3D)
            
            valid_count = tf.math.count_nonzero(tf.math.less_equal(error, threshold))
            count += valid_count.numpy()
        
        percentage[i] = count / (len(os.listdir(rhd_eval_dir)) * 21)
        i += 1
    
    thresholds = tf.stack(thresholds)
    thresholds = thresholds[:, 0] * 1000
    print(thresholds)
    pck_graph = plt.axes()
    pck_graph.grid()
    pck_graph.plot(thresholds, percentage, label = 'Our model')
    plt.title('RHD Dataset 3D PCK Curve')
    plt.xlabel('Error Thresholds (mm)')
    plt.ylabel('Percentage of Correct Keypoints')
    plt.show()
    

# checkpoint_dir is a directory that contains the model checkpoints to iterate over.
# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory that contains every single evaluation image (of just single hands).
# download_image is a function that we can call (we pass it one param: fileName), and 
#   it will return to us a numpy array for the image.
# y_test is a numpy array with the 3D keypoints for every single image in the RHD evaluation set.
#   the indices into this array are the names of the image files.
def generate_loss_graph(checkpoint_dir, model, rhd_eval_dir, download_image, y_test):
    pass


