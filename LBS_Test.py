# Authored by Lucas Coster
# Test for the MANO Linear Blend Skinning functions

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
import pickle
import time

mano_dir = "mano_v1_2"
file_path = os.path.join("..", mano_dir, "models", "MANO_RIGHT.pkl")
manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')
F = np.array(manoRight['f'], dtype=np.int32)

def lbs(beta, pose, J_, P_, K_, W_, S_, V_):

    #batch size
    bs = beta.size[0]

    #get device type
    device=beta.device

    #shape contribution
    v_shaped = V_ + blend_shape(beta, S_)

    #rest posed locations 
    j_rest = vertices2joints(v_shaped, J_)

    #add pose blend
    #3 by 1 axis angle to 3 by 3 rotation matrix
    eye3 = tf.eye(3,dtype=tf.float32,device=device)
    rmatrix = rodrigues(pose.view(-1,3).view(bs,-1,3,3))

    #pose feature
    pose_feature = (rmatrix[:, 1:, :, :] - eye3).view([bs, -1])
    #pose offsets
    pose_offsets = tf.matmul(pose_feature, P_).view(bs,-1,3)
    v_posed = v_shaped + pose_offsets

    #get global joint location
    j_transformed, A = batch_rigid_transform(rmatrix, j_rest, K_)

    #do skinning
    W = W_.expand([bs, -1, -1])
    T = tf.matmul(W, A.view(bs, -1, 16)).view(bs, -1, 4, 4)

    ones = tf.ones([bs, v_posed.shape[1],1], dtype=tf.float32, device=device)
    v_posed_homo = tf.concat([v_posed, ones], dim=2)
    v_homo = tf.matmul(T , tf.expand_dims(v_posed_homo, dim = -1))

    vertices = v_homo[:, :, :3, 0]

    return vertices, j_transformed

def vertices2joints(vert, J_):

    return tf.einsum ('bvt,jv->bjt', vert, J_)

def blend_shape(beta, S_):

    return tf.einsum ('bvt,jv->bjt', beta, S_)

def rodrigues(rvector):
    batch_size = rvector.shape[0]
    angle = tf.norm(rvector, axis = 1, keepdims= True)

    sin = tf.expand_dims(tf.math.sin(angle), axis = 2)
    cos = tf.expand_dims(tf.math.cos(angle), axis = 2)
   
    runit = rvector/angle
    kx, ky, kz = tf.split(runit, 3, axis = 1)
    zero = tf.zeros((batch_size,1))
    K = tf.reshape(tf.concat([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis = 1),[batch_size, 3, 3])
    I = tf.eye(3,batch_shape = [batch_size])
    rotationMatrix = I + sin * K + (1 - cos) * tf.linalg.matmul(K, K)    
    
    return rotationMatrix

def transform_matrix(R, t):
    return tf.concat ([F.pad(R, [0,0,0,1]), F.pad(t, [0,0,0,1], value=1)],dim=2)

def batch_rigid_transform(rmats, joints, parents):
    #applies a batch of rigid transforms to the joints
    #input rotation matrices
    #input joint locations
    #input kinematic trees of each object

    #output joint locations after applying the pose rotations
    #output transforms with respect to root joints

    joints = tf.expand_dims(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_matrix = transform_matrix(rmats.view(-1, 3, 3), rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_matrix[:,0]]

    for i in range(1, parents.shape[0]):
        #subtract joint location at rest pose
        curr_res = tf.matmul(transform_chain[parents[i]],transforms_matrix[:,i])
        transform_chain.append(curr_res)

    transforms = tf.stack(transform_chain, dim=1)

    #the last coloumn of the transfer contains the posed joints
    posed_joints = transforms[:,:,3,3]

    joints_homogen = F.pad(joints,[0,0,0,1])
    rel_transforms = transforms - F.pad(tf.matmul(transforms, joints_homogen), [3,0,0,0,0,0,0,0])

    return posed_joints, rel_transforms