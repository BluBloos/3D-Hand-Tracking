# TODO: Once the unit test is complete, update this file with the appropriate code.

import tensorflow as tf

from mano_bp import rmatrix_from_pose

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
    # rmats = tf.eye(3, num_columns=3, batch_shape=[bs, 16], dtype=tf.float32)

    ### BATCH_RIGID_TRANFORM ###
    # j_transformed, A = batch_rigid_transform(rmatrix, j_rest, K_)
    joints = tf.expand_dims(J_rest, axis =-1) # converts J_rest into a batch of column vectors.
    parents = K
    # TODO: Investigate if this is the right way to handle the root joint...
    #parents -= tf.concat([ tf.constant([4294967295], dtype=tf.int64), tf.zeros( (15), dtype=tf.int64 ) ], axis=0)
    
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
    