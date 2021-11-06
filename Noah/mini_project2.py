# Need to read in the RGB images and segmentation masks
# Need to mode the segmentation masks to be just black and white
# Need to write a quick "find the amount of islands" algo to reject all training and testing examples where there are two hands in the photo

# Have an intermediate loss term for learning the segmentaton masks.
# Then we have a final loss term for the full 21 joints prediction.

# thinking right of the bat that we convert the ground truth keypoints to gaussian heatmaps. This seems like it would be easier to learn...
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
    #fig = plt.figure(1)
    #ax1 = fig.add_subplot('221')
    #ax2 = fig.add_subplot('222')
    #ax3 = fig.add_subplot('223')
    #ax4 = fig.add_subplot('224', projection='3d')
    plt.imshow(image)
    plt.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    #ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
    #ax2.imshow(depth)
    #ax3.imshow(mask)
    #ax4.scatter(kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
    #ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    #ax4.set_xlabel('x')
    #ax4.set_ylabel('y')
    #ax4.set_zlabel('z')
    plt.show()

    break # Close program after just one sample (we are unit testing after all)

# UNIT TEST #1.
# Read in just one training example and plot the example data with the 2D keypoints right on top.
# DONE.


