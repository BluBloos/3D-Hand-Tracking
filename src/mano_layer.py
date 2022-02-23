# Author: Noah Cabral
# NOTE: This file is directly converted from the Pytorch implementation
# https://github.com/gmntu/mobilehand

# TODO: We currently only understand the MANO layer on a high-level.
# We need to work to understand how this works by getting into the details.

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
import os

# NOTE(Noah): Translation complete.
def vertices2joints(vert, J_):
  ''' 
  Calculates 3D joint positions from vertices
  using joint regressor array
  Input:
      vert [b,v,t] (batch size, num of vert, 3)
      J_   [j,v]   (num of joint, number of vert)
  Output:
      j_rest [b,j,t] (batch size, num of joint, 3)
  '''
  # TODO(Noah): Probably need to change what the equation is here.
  return tf.einsum('bvt,jv->bjt', [vert, J_])

# NOTE(Noah): Translation complete.
def blend_shape(beta, S_):
  ''' 
  Calculates per vertex displacement due to shape deformation
  i.e. Multiply each shape displacement (shape blend shape) by 
        its corresponding beta and then sum them
  Displacement [b,v,t] = sum_{l} beta[b,l] * S_[v,t,l]
  Input:
      beta [b,l]   (batch size, length=10)
      S_   [v,t,l] (num of vert, 3, length=10)
  Output:
      blend_shape [b,v,t] (batchsize, num of vert, 3)
  '''
  # TODO(Noah): Once more, it is probably the case that we need to change
  # the eq here.
  return tf.einsum('bl,vtl->bvt', [beta, S_])

# NOTE(Noah): Translation complete.
def batch_rodrigues(rvecs, epsilon=1e-8):
  ''' 
  Calculates the rotation matrices for a batch of rotation vectors
  Input:
      rvecs [N,3] array of N axis-angle vectors
  Output:
      rmat [N,3,3] rotation matrices for the given axis-angle parameters
  '''
  # Get batch size
  bs = rvecs.shape[0]
  # Get device type
  #device = rvecs.device
  #angle = torch.norm(rvecs + 1e-8, dim=1, keepdim=True)
  angle = tf.norm(rvecs + 1e-8, axis=1, keepdims=True)
  rot_dir = rvecs / angle
  # NOTE(Noah): Unsure if tf.cos / tf.sin is sensible.
  #cos = torch.unsqueeze(torch.cos(angle), dim=1)
  cos = tf.expand_dims(tf.cos(angle), axis=1)
  sin = tf.expand_dims(tf.sin(angle), axis=1)
  # Bx1 arrays
  #rx, ry, rz = torch.split(rot_dir, 1, dim=1)
  rx, ry, rz = tf.split(rot_dir, 1, axis=1)
  #K = torch.zeros((bs, 3, 3), dtype=torch.float32, device=device)
  K = tf.zeros((bs, 3, 3), dtype=tf.float32)
  #zeros = torch.zeros((bs, 1), dtype=torch.float32, device=device)
  zeros = tf.zeros((bs,1), dtype=tf.float32)
  #K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
  #    .view((bs, 3, 3))
  K = tf.reshape(
      tf.concat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1),
      (bs, 3, 3)
  )
  #eye3 = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(dim=0)
  eye3 = tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0)
  #rmats = eye3 + sin * K + (1 - cos) * torch.bmm(K, K)
  rmats = eye3 + sin * K + (1 - cos) * tf.linalg.matmul(K, K)
  return rmats

# TODO(Noah): It seems that the calling of tf.pad here is not going to do it.
# Of course, we only know this after our experience with tf.pad today.
def transform_mat(R, t):
  ''' 
  Creates a batch of transformation matrices
  
  Input:
      R [B,3,3] array of a batch of rotation matrices
      t [B,3,1] array of a batch of translation vectors
  
  Output
      T [B,4,4] transformation matrix
  '''
  # No padding left or right, only add an extra row
  # return torch.cat([F.pad(R, [0, 0, 0, 1]),
  #                   F.pad(t, [0, 0, 0, 1], value=1)], dim=2)
  return tf.concat(
      [tf.pad(R, tf.constant([[0,0],[0,1]])),
      tf.pad(t, tf.constant([[0,0],[0,1]]), constant_values=1)],
      axis=2
  )

def batch_rigid_transform(rmats, joints, parents):
  '''
  Applies a batch of rigid transformations to the joints
  
  Input:
      rmats   [B,N,3,3] rotation matrices
      joints  [B,N,3]   joint locations
      parents [B,N]     kinematic tree of each object
  Output:
      posed_joints   [B,N,3] 
          joint locations after applying the pose rotations
      rel_transforms [B,N,4,4] 
          relative (with respect to the root joint) 
          rigid transformations for all the joints
  '''
  #joints = torch.unsqueeze(joints, dim=-1)
  joints = tf.expand_dims(joints, axis=1)

  #rel_joints = joints.clone() # [1, 16, 3, 1]
  rel_joints = tf.identity(joints)
  rel_joints[:, 1:] -= joints[:, parents[1:]]

  #transforms_mat = transform_mat(
  #    rmats.view(-1, 3, 3),
  #    rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

  # TODO(Noah): I reckon tf.shape is going to do some odd things...
  transforms_mat = transform_mat(
      tf.reshape(rmats, (-1, 3, 3)),
      tf.reshape(tf.reshape(rel_joints, (-1, 3, 1)), (-1, tf.shape(joints), 4, 4) )
  )

  transform_chain = [transforms_mat[:, 0]]
  for i in range(1, parents.shape[0]):
      # Subtract the joint location at the rest pose
      # No need for rotation, since it's identity when at rest
      # curr_res = torch.matmul(transform_chain[parents[i]],
      #                        transforms_mat[:, i])
      curr_res = tf.linalg.matmul(transform_chain[parents[i]], transforms_mat[:, i])
      transform_chain.append(curr_res)

  transforms = tf.stack(transform_chain, axis=1)

  # The last column of the transformations contains the posed joints
  posed_joints = transforms[:, :, :3, 3]

  joints_homogen = tf.pad(joints, tf.constant([[0, 0], [0, 1]]))

  rel_transforms = transforms - tf.pad(
    tf.linalg.matmul(transforms, joints_homogen), 
    tf.constant([ [3, 0], [0, 0], [0, 0], [0, 0]]))

  return posed_joints, rel_transforms

def lbs(beta, pose, V_, K_, S_, P_, J_, W_):
  # Get batch size
  bs = tf.shape(beta)[0]

  # Add shape contribution
  v_shaped = V_ + blend_shape(beta, S_) # [bs, 778, 3]

  # Get rest posed joints locations
  # NxJx3 array
  #j_rest = vertices2joints(v_shaped, J_).contiguous() # [bs, 16, 3]
  j_rest = vertices2joints(v_shaped, J_) # [bs, 16, 3]

  # Add pose blend shapes
  # To convert 3 by 1 axis angle to 3 by 3 rotation matrix
  # Note: pose [bs, 16, 3] reshape to [bs*16, 3] for batch rodrigues
  #eye3 = torch.eye(3, dtype=torch.float32, device=device) # Identity matrix
  eye3 = tf.eye(3, dtype=tf.float32)
  # rmats = batch_rodrigues(pose.view(-1, 3)).view(bs, -1, 3, 3) # [bs, 16, 3, 3]
  # pose_feature [bs, 135] where 135 = 15*9
  # pose_feature = (rmats[:, 1:, :, :] - eye3).view([bs, -1])
  # pose_offsets [bs, 135] matmul [135, 778*3] = [bs, 778*3] -> [bs, 778, 3]
  # pose_offsets = torch.matmul(pose_feature, P_).view(bs, -1, 3)
  # v_posed      = v_shaped + pose_offsets # [bs, 778, 3]

  # Get global joint location
  # j_transformed [bs, 16, 3], A [bs, 16, 4, 4]
  #j_transformed, A = batch_rigid_transform(rmats, j_rest, K_) 

  # Do skinning
  #W = W_.unsqueeze(dim=0).expand([bs, -1, -1]) # [bs, 778, 16]
  # W[bs, 778, 16] matmul A[bs, 16, 16] = [bs, 778, 16] 
  # -> reshape to [bs, 778, 4, 4]
  #T = torch.matmul(W, A.view(bs, -1, 16)).view(bs, -1, 4, 4) # [bs, 778, 4, 4]

  #ones = torch.ones([bs, v_posed.shape[1], 1], 
  #    dtype=torch.float32, device=device) # [bs, 778, 1]
  #v_posed_homo = torch.cat([v_posed, ones], dim=2) # [bs, 778, 4]
  # T[bs, 778, 4, 4] matmul v_posed_homo unsqueeze [bs, 778, 4, 1]
  #v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1)) # [bs, 778, 4, 1]

  #vertices = v_homo[:, :, :3, 0] # [bs, 778, 3]

  #return vertices, j_transformed
  pass


import pickle

# NOTE(Noah): For the implementation of the MANO model, 
# we are heavily referencing https://github.com/gmntu/mobilehand/blob/master/code/utils_mpi_model.py

# Noah + Maddie

class MANO_Model(Model):
  
  # TODO(Noah): We probably need to extend some sort of class from Tensorflow
  # to make a layer.
  def __init__(self, mano_dir, verbose=False):
    super(MANO_Model, self).__init__()
    
    # --- Load in learned mano paramaters ---
    file_path = os.path.join(mano_dir, "models", "MANO_RIGHT.pkl")
    
    try:
      manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')

      self.V = manoRight['v_template']          # Vertices of template model (V), Shape=(778, 3), type=uint32
      self.F = manoRight['f']                   # Faces of the model (F), Shape=(1538, 3), type=uint32
      self.K = manoRight['kintree_table'][0]   # Kinematic tree defining the parent joint (K), Shape=(16,), type=int64
      self.S = manoRight['shapedirs']           # Shape blend shapes that are learned (S), Shape=(778, 3, 10), type=float64
      self.P = manoRight['posedirs']            # Pose blend shapes that are learned (P), Shape=(778, 3, 135), type=float64
      self.J = manoRight['J_regressor']         # Joint regressor that are learned (J), Shape=(16,778), type=float64
      self.W = manoRight['weights']             # Weights that are learned (W), Shape=(778,16), type=float64
      self.C = manoRight['hands_components']    # Components of hand PCA (C), Shape=(45, 45), type=float64
      self.M = manoRight['hands_mean']          # Mean hand PCA pose (M), Shape=(45,), type=float64

      # Convert loaded params to Numpy
      self.V = np.array(self.V, dtype=np.float32)
      self.F = np.array(self.F, dtype=np.int32) # Need to convert from uint32 to int32 to allow interation in v0 = vertices[:, self.F[:,0],:] # [bs, 1538, 3]    
      self.S = np.array(self.S, dtype=np.float32)
      self.P = np.array(self.P, dtype=np.float32)
      self.J = np.array(self.J.todense(), dtype=np.float32) # Need to convert sparse to dense matrix
      self.W = np.array(self.W, dtype=np.float32)
      self.C = np.array(self.C, dtype=np.float32) # Use PCA
      self.M = np.array(self.M, dtype=np.float32)
      self.K[0] = -1 # Convert undefined parent of root joint to -1
      
      # Reshape the pose blend shapes from (778, 3, 135) -> (778*3, 135) -> (135, 778*3)
      num_pose_basis = self.P.shape[-1]
      P = np.reshape(self.P, [-1, num_pose_basis]).T

      # Anatomy 101: https://en.wikipedia.org/wiki/Metacarpophalangeal_joint
      # MCP = your kunckle
      # PIP = the kunckle after your knuckle
      # DIP = just before the fingernail.

      '''
      NOTE(Noah): So here's what's going on here. MANO defines a standard convention for the order of the joints,
      Then the authors of MobileHand decided to switch things up.
      '''
      # [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] joint order (total 16 joints including wrist)
      # MANO joint convention:
      # Note: The order of ring and little finger is swapped!
      # T   I  M  R   L (Thumb, Index, Middle, Ring, Little)
      # 16 17 18  19 20 (Fingetip newly added)
      # 15  3  6  12  9 (DIP)
      # 14  2  5  11  8 (PIP)
      # 13  1  4  10  7 (MCP)
      #       0 (Wrist)

      '''NEW ORDER, and supposedly the standard convention?'''
      # Rearrange MANO joint convention to standard convention:
      #  0:Wrist,
      #  1:TMCP,  2:TPIP,  3:TDIP,  4:TTIP (Thumb)
      #  5:IMCP,  6:IPIP,  7:IDIP,  8:ITIP (Index)
      #  9:MMCP, 10:MPIP, 11:MDIP, 12:MTIP (Middle)
      # 13:RMCP, 14:RPIP, 15:RDIP, 16:RTIP (Ring)
      # 17:LMCP, 18:LPIP, 19:LDIP, 20:LTIP (Little)
      self.remap_joint = [ 0,          # Wrist
                          13,14,15,16, # Thumb
                            1, 2, 3,17, # Index
                            4, 5, 6,18, # Middle
                          10,11,12,19, # Ring
                            7, 8, 9,20] # Little
      

      # we note that zmat exists to map some smaller pose space to the 
      # space of 45 params that MANO expects.
      self.Z = self.generate_Zmat(self.S, self.V, self.J)

      #########################
      ### Define pose limit ###
      #########################
      # Values adapted from
      # HOnnotate: A method for 3D Annotation of Hand and Object Poses Supplementary Material
      # https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/team_lepetit/images/hampali/supplementary.pdf
      self.plim = np.array([
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
          [[ 0.00, 0.00], [-0.50, 0.00], [-1.57, 1.08]]])# DIP

      ################################
      ### Define joint angle limit ###
      ################################
      self.alim = np.array([
          # MCP a/a, MCP f/e, PIP f/e, DIP f/e
          [-20,30], [-10,90], [-1,90], [-1,90], # Index [min,max]
          [-20,10], [-10,90], [-1,90], [-1,90], # Middle
          [-20,10], [-10,90], [-1,90], [-1,90], # Ring
          [-30,20], [-10,90], [-1,90], [-1,90], # Little
          # x-axis, y-axis  , z-axis
          # [-45,140], [-45,45], [-45,45], # Thumb TM
          [-45,45], [-45,45], [-45,45], # Thumb TM
          [-45, 45], [-45,45], [-45,45], # Thumb MCP
          [-1 , 90]])                    # Thumb IP flex/ext
      # Convert degrees to radians
      self.alim = np.radians(self.alim)

      # self.ReLU = 

      print('MANO Differentiable Layer Loaded')

    except:
      print("Unable to find MANO_RIGHT.pkl")

  # NOTE(Noah): I have absolutely no idea how this works. But, I am just going to take it as
  # it is, because this is just pure numpy, Python, and opencv code. No PyTorch in sight here.
  def generate_Zmat(self, S, V, J):
    Z = np.zeros((23, 45))

    # Note: MANO pose has a total of 15 joints
    #       Each joint 3 DoFs thus pose has a total 15*3 = 45 values
    # But actual human hand only has a total of 21/22/23 DoFs
    # (21 DoFs for 4 fingers(4x4) + 1 thumb(5/6/7))
    # Thus joint angle will have 23 values (using thumb with 7 DoFs)

    #######################################
    ### Get the joints at rest position ###
    #######################################
    Bs = S.dot(np.zeros(10)) # (778, 3, 10) dot (10) = (788, 3)
    Vs = V + Bs              # (788, 3) Vertices of template (V) are modified in an additive way by adding Bs
    Jrest = J.dot(Vs)        # (16, 778) dot (778, 3) = (16, 3) Rest joint locations

    ###################################
    ### Create some lamda functions ###
    ###################################
    # Convert rotation vector (3 by 1 or 1 by 3) to rotation matrix (3 by 3)
    rvec2rmat = lambda rvec : cv2.Rodrigues(rvec)[0]
    # Note: j1 is the joint of interest and j2 is its parent (1 by 3)
    rotate_finger = lambda j1, j2 : rvec2rmat(np.array(
      [0,np.arctan((j1[2]-j2[2])/(j1[0]-j2[0])),0])) # Rotate about y axis
    # Note: Thumb is rotated by around some degrees relative to the rest of the fingers
    # Note: pL is the left and pR is the right point of thumb fingernail (1 by 3)
    rotate_thumb = lambda pL, pR : rvec2rmat(np.array(
      [np.arctan((pL[1]-pR[1])/(pL[2]-pR[2])),0,0])) # Rotate about x axis

    ####################
    ### Index finger ###
    ####################
    Z[0:2,0:3] = rotate_finger(Jrest[1,:],Jrest[0,:])[1:3,:] # 0:MCP abduct/adduct, 1:MCP flex/ext
    Z[  2,3:6] = rotate_finger(Jrest[2,:],Jrest[1,:])[  2,:] # 2:PIP flex/ext
    Z[  3,6:9] = rotate_finger(Jrest[3,:],Jrest[2,:])[  2,:] # 3:DIP flex/ext
    #####################
    ### Middle finger ###
    #####################
    Z[4:6, 9:12] = rotate_finger(Jrest[4,:],Jrest[0,:])[1:3,:] # 4:MCP abduct/adduct, 5:MCP flex/ext
    Z[  6,12:15] = rotate_finger(Jrest[5,:],Jrest[4,:])[  2,:] # 6:PIP flex/ext
    Z[  7,15:18] = rotate_finger(Jrest[6,:],Jrest[5,:])[  2,:] # 7:DIP flex/ext
    ###################
    ### Ring finger ###
    ###################
    Z[8:10,27:30] = rotate_finger(Jrest[10,:],Jrest[ 0,:])[1:3,:] # 8:MCP abduct/adduct, 9:MCP flex/ext
    Z[  10,30:33] = rotate_finger(Jrest[11,:],Jrest[10,:])[  2,:] # 10:PIP flex/ext
    Z[  11,33:36] = rotate_finger(Jrest[12,:],Jrest[11,:])[  2,:] # 11:DIP flex/ext
    #####################
    ### Little finger ###
    #####################
    Z[12:14,18:21] = rotate_finger(Jrest[7,:],Jrest[0,:])[1:3,:] # 12:MCP abduct/adduct, 13:MCP flex/ext
    Z[   14,21:24] = rotate_finger(Jrest[8,:],Jrest[7,:])[  2,:] # 14:PIP flex/ext
    Z[   15,24:27] = rotate_finger(Jrest[9,:],Jrest[8,:])[  2,:] # 15:DIP flex/ext
    #############
    ### Thumb ###
    #############
    thumb_left, thumb_right = 747, 720
    Z[16:19,36:39] = np.eye(3) # 16:TM  rx, 17:TM  ry, 18:TM  rz
    Z[19:22,39:42] = np.eye(3) # 19:MCP rx, 20:MCP ry, 21:MCP rz
    Z[   22,42:45] = rotate_thumb(Vs[thumb_left,:],Vs[thumb_right,:]).dot(
        rotate_finger(Jrest[15,:],Jrest[14,:]))[2,:] # 22:IP flex/ext

    return Z

  def call(self, x, training=False):
    return x