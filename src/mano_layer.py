# Author: 

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
import os
import pickle

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

      print('MANO Differentiable Layer Loaded')

    except:
      print("Unable to find MANO_RIGHT.pkl")

  def call(self, beta, pose, root_trans, training=False):

    T_shaped = self.T_bar + blend_shape(beta, self.S)
    B_p = blend_pose(pose, self.P)

    posed_joints, mesh = lbs(pose, self.J, self.K, self.W, T_shaped, B_p)

    tips = tf.gather(mesh, indices=[745, 333, 444, 555, 672], axis=1)
    joints_full = tf.concat( [posed_joints, tips], axis=1)

    mesh += root_trans
    joints_full += root_trans
 
    return mesh, joints_full
