# Author(s): Noah Cabral, Lucas Coster, Max Vincent

import tensorflow as tf
from tensorflow.keras import Model
import os
import pickle

def vertices2joints(vert, J_):
    return tf.einsum('bvt,jv->bjt', vert, J_)

def transform_matrix(R, t):
    bs = R.shape[0]
    T = tf.concat([R,t], axis=2) # 3x4 (3 rows, 4 columns)
    row = tf.constant([[0,0,0,1]], dtype = tf.float32)
    row_batched = tf.repeat(tf.expand_dims(row, axis=0), repeats=[bs], axis=0)
    T = tf.concat([T,row_batched], axis = 1)
    return T

def lbs(pose, J, K, W, T_shaped, B_p):
    bs = pose.shape[0]
    J_rest = vertices2joints(T_shaped, J)
    rmats = rmatrix_from_pose(pose)

    ### BATCH_RIGID_TRANFORM ###
    joints = tf.expand_dims(J_rest, axis =-1) # converts J_rest into a batch of column vectors.
    parents = K
    
    # rel_joints is defined such that it stores the location of all joints in the coordinate space
    # of their ancestor. This is as oposed to joints which stores the location of all joints in the
    # global coordinate space.
    rel_joints = tf.identity(joints)
    rel_joints -= tf.concat( 
        [ tf.zeros([bs, 1, 3, 1]), tf.gather(joints[:, :], indices=parents[1:], axis=1) ],
        axis=1
    )
    tm = transform_matrix(tf.reshape(rmats, (-1, 3, 3)), tf.reshape(rel_joints, (-1, 3, 1)))

    # We note that tm gets collapsed along the batch dimension (combining batches with 2nd dim)
    # So we transform it back.
    tm = tf.reshape(tm, ((-1, joints.shape[1], 4, 4)))
  
    # So this is Gk as seen in SMPL paper but as a Python list.
    #
    # What happens in the formulation of Gk here is that the joints array is defined
    # such that for any ith joint in the list, the parent will have some index i_prime that
    # is less than i. So the parents always come first.
    # otherwise we would get indexing errors into the code below.
    #
    # and if we think for a moment about what the code below is doing?
    # it is building the Gk for each k part, and it is does this by continually
    # climbing up the ancestor chain.
    Gk_pylist = [ tm[:,0] ]
    for i in range(1, parents.shape[0]):
        Gk_pylist.append( tf.matmul(Gk_pylist[parents[i]], tm[:,i]) )
    Gk = tf.stack(Gk_pylist, axis=1)
   
    # Why is it that these are actually the posed_joints?
    # because the joints are already in their own ref frame.
    posed_joints = tf.slice( Gk[:, :, :, 3], [0, 0, 0], [bs, 16, 3])
  
    # Remove the rest pose from each Gk.
    rmat_rest = rmatrix_from_pose(tf.zeros([bs, 16, 3]))
    
    # NOTE(Noah): Maybe we want to change transform matrix to stop being so silly...?
    Gk_rest_inv = transform_matrix(tf.reshape(rmat_rest, [-1, 3, 3]), 
        -tf.reshape(joints, [-1,3,1]))
    Gk_rest_inv = tf.reshape(Gk_rest_inv, ((-1, joints.shape[1], 4, 4)))
   
    Gk_prime = tf.matmul(Gk, Gk_rest_inv) 

    W_batched = tf.repeat(tf.expand_dims(W, axis=0), repeats=[bs], axis=0)
    L = tf.matmul(W_batched, tf.reshape(Gk_prime, (bs, -1, 16)))
    L = tf.reshape(L, (bs, -1, 4, 4))

    _T = T_shaped + B_p
    ones = tf.ones([bs, _T.shape[1], 1], dtype=tf.float32)
    _T_homo = tf.concat([_T, ones], axis=2)
    T_prime = tf.matmul(L, tf.expand_dims(_T_homo, axis=-1))
    mesh = T_prime[:, :, :3, 0]

    return posed_joints, mesh

def blend_shape(beta, S):
  # we note that l is the number of beta parameters.
  # v is the number of vertices in the MANO hand (778)
  # t is the dimensionality of each vertex (3).
  # b is the batch_size.
  return tf.einsum('bl,vtl->bvt', beta, S)

def rodrigues(rvector):
    batch_size = rvector.shape[0]
    angle = tf.norm(rvector, axis = 1, keepdims= True)
    value = tf.fill((batch_size,1), 1e-8)
    angle = tf.math.add(angle, value)

    sin = tf.expand_dims(tf.math.sin(angle), axis = 2)
    cos = tf.expand_dims(tf.math.cos(angle), axis = 2)
   
    runit = rvector/angle
    
    kx, ky, kz = tf.split(runit, 3, axis = 1)
    zero = tf.zeros((batch_size,1))
    K = tf.reshape(tf.concat([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis = 1),[batch_size, 3, 3])
    I = tf.eye(3, batch_shape = [batch_size])
    rotationMatrix = I + sin * K + (1 - cos) * tf.linalg.matmul(K, K) 
    return rotationMatrix

def rmatrix_from_pose(pose):
    batch_size = pose.shape[0]
    return tf.reshape(rodrigues(tf.reshape(pose,(-1, 3))), (batch_size, -1, 3, 3))

def blend_pose(pose, P):
    batch_size = pose.shape[0]
    I = tf.eye(3)
    rotationMatrix = rmatrix_from_pose(pose)
    pose_sum = tf.reshape(rotationMatrix[:, 1:, :, :]-I, (batch_size, -1))
    blend_pose = tf.reshape(tf.linalg.matmul(pose_sum, tf.reshape(P, (135, -1))), (batch_size, -1, 3))
    return blend_pose

class MANO_Model(Model):
  
  def __init__(self, mano_dir, **kwargs):
    super().__init__(**kwargs)
    
    # --- Load in learned MANO paramaters ---
    file_path = os.path.join(mano_dir, "models", "MANO_RIGHT.pkl")
    manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')

    self.T_bar = manoRight['v_template']      # Vertices of template model (V), Shape=(778, 3), type=uint32
    self.F = manoRight['f']                   # Faces of the model (F), Shape=(1538, 3), type=uint32
    self.K = manoRight['kintree_table'][0]    # Kinematic tree defining the parent joint (K), Shape=(16,), type=int64
    self.S = manoRight['shapedirs']           # Shape blend shapes that are learned (S), Shape=(778, 3, 10), type=float64
    self.P = manoRight['posedirs']            # Pose blend shapes that are learned (P), Shape=(778, 3, 135), type=float64
    self.J = manoRight['J_regressor']         # Joint regressor that are learned (J), Shape=(16,778), type=float64
    self.W = manoRight['weights']             # Weights that are learned (W), Shape=(778,16), type=float64
    
    # Convert loaded params to Numpy arrays.
    self.T_bar = tf.convert_to_tensor(self.T_bar, dtype=tf.float32)
    # NOTE(Noah): Need to convert from uint32 to int32 to allow interation in 
    #   v0 = vertices[:, self.F[:,0],:] # [bs, 1538, 3]  
    self.F = tf.convert_to_tensor(self.F, dtype=tf.int32)   
    self.S = tf.convert_to_tensor(self.S, dtype=tf.float32)
    self.P = tf.convert_to_tensor(self.P, dtype=tf.float32)
    self.J = tf.convert_to_tensor(self.J.todense(), dtype=tf.float32) # Need to convert sparse to dense matrix
    self.W = tf.convert_to_tensor(self.W, dtype=tf.float32)

    # move T_Bar such that it is at (0, 0, 0)
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
    
    # tip to palm.

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
    # These specific values were sourced from the supplementary material of 
    # HOnnotate: A method for 3D Annotation of Hand and Object Poses
    # -> https://arxiv.org/abs/1907.01481
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

    # Like, maybe this does not reshape properly down...?

    self.L = tf.expand_dims(plim[:, :, 0], axis=0) # [1, 15, 3]
    self.U = tf.expand_dims(plim[:, :, 1], axis=0) # [1, 15, 3]

    #self.L = tf.expand_dims(tf.reshape(plim[:, :, 0], shape=(45)), axis=0)
    #self.U = tf.expand_dims(tf.reshape(, shape=(45)), axis=0)

    print('MANO Differentiable Layer Loaded')

  def call(self, beta, pose, training=False):
    T_shaped = self.T_bar + blend_shape(beta, self.S)
    B_p = blend_pose(pose, self.P)

    posed_joints, posed_mesh = lbs(pose, self.J, self.K, self.W, T_shaped, B_p)

    # Add fingertips and remap to RHD convention.
    indices = [745, 333, 444, 555, 672]
    fingertips = tf.gather(posed_mesh, indices, axis=1)
    posed_joints = tf.concat( [posed_joints, fingertips], axis=1)
    posed_joints = tf.gather(posed_joints, indices=self.remap_joints, axis=1)
 
    return posed_mesh, posed_joints
