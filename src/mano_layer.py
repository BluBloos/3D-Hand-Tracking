# Author: 

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
import os
import pickle
from qmindcolors import cstr

from mano_bs import blend_shape
from mano_bp import blend_pose
from mano_lbs import lbs

class MANO_Model(Model):
  
  def __init__(self, mano_dir):
    super(MANO_Model, self).__init__()
    
    # --- Load in learned mano paramaters ---
    file_path = os.path.join(mano_dir, "models", "MANO_RIGHT.pkl")
    
    try:
      manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')

      self.T_bar = manoRight['v_template']          # Vertices of template model (V), Shape=(778, 3), type=uint32
      self.F = manoRight['f']                   # Faces of the model (F), Shape=(1538, 3), type=uint32
      self.K = manoRight['kintree_table'][0]    # Kinematic tree defining the parent joint (K), Shape=(16,), type=int64
      self.S = manoRight['shapedirs']           # Shape blend shapes that are learned (S), Shape=(778, 3, 10), type=float64
      self.P = manoRight['posedirs']            # Pose blend shapes that are learned (P), Shape=(778, 3, 135), type=float64
      self.J = manoRight['J_regressor']         # Joint regressor that are learned (J), Shape=(16,778), type=float64
      self.W = manoRight['weights']             # Weights that are learned (W), Shape=(778,16), type=float64
     
      # Convert loaded params to Numpy arrays.
      self.T_bar = tf.convert_to_tensor(self.T_bar, dtype=tf.float32)
      self.F = tf.convert_to_tensor(self.F, dtype=tf.int32) # Need to convert from uint32 to int32 to allow interation in v0 = vertices[:, self.F[:,0],:] # [bs, 1538, 3]    
      self.S = tf.convert_to_tensor(self.S, dtype=tf.float32)
      self.P = tf.convert_to_tensor(self.P, dtype=tf.float32)
      self.J = tf.convert_to_tensor(self.J.todense(), dtype=tf.float32) # Need to convert sparse to dense matrix
      self.W = tf.convert_to_tensor(self.W, dtype=tf.float32)

      # move T_Bar such that it is at (0,0,0)
      root_pos = tf.constant([[[0.09566993,  0.00638343,  0.00618631]]])
      self.T_bar -= root_pos

      # Apply a rotation to make the MANO hand be in a more sensible default position.
      batch_size = 1
      beta = tf.zeros([batch_size, 10])
      pose = tf.repeat(tf.constant([[
          [ 
            # NOTE: this rotation in axis-angle representation was 
            # determined by the handy https://www.andre-gaschler.com/rotationconverter/
            0.5773503 * 2.0943951, 
            0.5773503 * 2.0943951, 
            0.5773503 * 2.0943951
          ], [0,0,0], [0,0,0], 
          [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0],
          [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]
      ]], dtype=tf.float32), repeats=[batch_size], axis=0)
      B_p = blend_pose(pose, self.P)
      posed_joints, posed_mesh = lbs(pose, self.J, self.K, self.W, self.T_bar, B_p)
      self.T_bar = posed_mesh[0] # Select just first batch.

      # indices are the RHD convention, the stored values are indices in MANO convention.
      self.remap_joints = tf.constant(
        [
          0,              # Wrist (0)
          16, 15, 14, 13, # Thumb (1, 2, 3, 4)
          17, 3, 2, 1,    # Index (5, 6, 7, 8)
          18, 6, 5, 4,    # Middle (9, 10, 11, 12)
          19, 12, 11, 10, # Ring (13, 14, 15, 16)
          20, 9, 8, 7     # Little (Pinky) (17, 18, 19, 20)
        ], 
        dtype=tf.int32
      )

      # Rendered hand pose is tip to palm.
      # What about the MANO? What is the representation that they use ??

      # Indices and stored values are all RHD convention.
      self.RHD_K = tf.constant(
        [
          -1, # root 
          2, 3, 4, 0,
          6, 7, 8, 0, 
          10, 11, 12, 0, 
          14, 15, 16, 0,
          18, 19, 20, 0
        ],
        dtype=tf.int32
      )

      # Indices are RHD convention, values in K are MANO convention.
      self.K_remaped = tf.gather(
        tf.concat([tf.constant(self.K, dtype=tf.int32), tf.constant([15, 3, 6, 12, 9])], axis=0),
        indices=self.remap_joints, axis=0
      )

      # MANO Convention.
      plim = tf.constant([
        # Index
        [[ 0.00, 0.45], [-0.15, 0.20], [0.10, 1.80]], # MCP
        [[-0.30, 0.20], [ 0.00, 0.00], [0.00, 0.20]], # PIP
        [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
        # Middle
        [[ 0.00, 0.00], [-0.15, 0.15], [0.10, 2.00]], # MCP
        [[-0.50,-0.20], [ 0.00, 0.00], [0.00, 2.00]], # PIP
        [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
        # Little
        [[-1.50,-0.20], [-0.15, 0.60], [-0.10, 1.60]], # MCP
        [[ 0.00, 0.00], [-0.50, 0.60], [ 0.00, 2.00]], # PIP
        [[ 0.00, 0.00], [ 0.00, 0.00], [ 0.00, 1.25]], # DIP
        # Ring
        [[-0.50,-0.40], [-0.25, 0.10], [0.10, 1.80]], # MCP
        [[-0.40,-0.20], [ 0.00, 0.00], [0.00, 2.00]], # PIP
        [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
        # Thumb
        [[ 0.00, 2.00], [-0.83, 0.66], [ 0.00, 0.50]], # MCP
        [[-0.15,-1.60], [ 0.00, 0.00], [ 0.00, 0.50]], # PIP
        [[ 0.00, 0.00], [-0.50, 0.00], [-1.57, 1.08]]]) # DIP

      self.L = tf.expand_dims(tf.reshape(plim[:, :, 0], shape=(45)), axis=0)
      self.U = tf.expand_dims(tf.reshape(plim[:, :, 1], shape=(45)), axis=0)

      print('MANO Differentiable Layer Loaded')

    except Exception as e:
      print(e, "Unable to find MANO_RIGHT.pkl")

  def call(self, beta, pose, scale, depth, training=False):

    bs = beta.shape[0]
    T_shaped = self.T_bar + blend_shape(beta, self.S)
    B_p = blend_pose(pose, self.P)

    posed_joints, posed_mesh = lbs(pose, self.J, self.K, self.W, T_shaped, B_p)

    # Add fingertips and remap to RHD convention.
    indices = [745, 333, 444, 555, 672]
    fingertips = tf.gather(posed_mesh, indices, axis=1)
    posed_joints = tf.concat( [posed_joints, fingertips], axis=1)
    posed_joints = tf.gather(posed_joints, indices=self.remap_joints, axis=1)

    # NOTE: We define the scale as the distance between the root of the hand
    # and the knuckle on the index finger. Passing into the model a scale of 1.0
    # ensures that this distance is exactly 0.1537328322252615.

    mano_scale = tf.math.sqrt(tf.reduce_sum(tf.math.square(posed_joints[:, 0] - posed_joints[:, 8]), axis=1))
    print(cstr("mano_scale"), mano_scale)
    mano_scale = tf.reshape(mano_scale, shape=[bs, 1, 1])
    scaling_factor = (tf.repeat(tf.constant([[[0.1537328322252615]]]), bs, axis=0) / mano_scale) * scale

    posed_mesh = posed_mesh * scaling_factor
    posed_joints = posed_joints * scaling_factor

    root_trans = tf.constant( [[ [0.0, 0.0, depth] ]] )
    posed_mesh += root_trans
    posed_joints += root_trans
 
    return posed_mesh, posed_joints
