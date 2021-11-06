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
    # plt.imshow(image)
    # plt.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    # plt.show()

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

    plt.imshow(mask)
    plt.show()

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


