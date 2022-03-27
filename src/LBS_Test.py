# Authored by Lucas Coster
# Test for the MANO Linear Blend Skinning functions

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
import pickle
import time

from rodrigues import rodrigues

def lbs(pose, J_, K_, W_, V_):

    #batch size
    bs = pose.shape[0]

    #shape contribution
    v_shaped = V_ # + blend_shape(beta, S_)

    #rest posed locations 
    j_rest = vertices2joints(v_shaped, J_)

    #add pose blend
    #3 by 1 axis angle to 3 by 3 rotation matrix
    # eye3 = tf.eye(3, dtype=tf.float32)
    rmatrix = rodrigues(pose.view(-1,3).view(bs,-1,3,3))

    #pose feature
    # pose_feature = (rmatrix[:, 1:, :, :] - eye3).view([bs, -1])
    #pose offsets
    #pose_offsets = tf.matmul(pose_feature, P_).view(bs,-1,3)
    v_posed = v_shaped # + pose_offsets

    #get global joint location
    j_transformed, A = batch_rigid_transform(rmatrix, j_rest, K_)

    # do skinning
    W = W_.expand([bs, -1, -1])
    T = tf.matmul(W, A.view(bs, -1, 16)).view(bs, -1, 4, 4)

    ones = tf.ones([bs, v_posed.shape[1],1], dtype=tf.float32)
    v_posed_homo = tf.concat([v_posed, ones], dim=2)
    v_homo = tf.matmul(T , tf.expand_dims(v_posed_homo, dim = -1))

    vertices = v_homo[:, :, :3, 0]

    return vertices, j_transformed

def vertices2joints(vert, J_):

    return tf.einsum ('bvt,jv->bjt', vert, J_)

def blend_shape(beta, S_):

    return tf.einsum ('bvt,jv->bjt', beta, S_)

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

import open3d as o3d

if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    manoRight = None

    try:
        # Here we load in the learned MANO parameters.
        mano_dir = "mano_v1_2"
        file_path = os.path.join("..", mano_dir, "models", "MANO_RIGHT.pkl")
        manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')
    except Exception as e:
        print("Oops! Something went wrong.\n\
        It's likely that MANO_RIGHT.pkl was not found.\n\
        Check that this file exists in mano_v1_2/models")
        print(e)

    if manoRight != None:

        S = tf.convert_to_tensor(manoRight['shapedirs'], dtype=tf.float32) # shape of S is (778, 3, 10)
        T_bar = tf.convert_to_tensor(manoRight['v_template'], dtype=tf.float32) # shape of (778, 3)
        F = np.array(manoRight['f'], dtype=np.int32)  

        # Kinematic tree defining the parent joint (K), Shape=(16,), type=int64
        K = np.array(manoRight['kintree_table'][0], dtype=np.int64)
        # Joint regressor that are learned (J), Shape=(16,778), type=float64   
        J = np.array(manoRight['J_regressor'].todense(), dtype=np.float64)
        # Weights that are learned (W), Shape=(778,16), type=float64       
        W = np.array(manoRight['weights'], dtype=np.float64)             
        
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # create the meshes
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(mesh_frame)  
        meshes = []
        
        for i in range(5):
          mesh = o3d.geometry.TriangleMesh()
          mesh_t = (0.1 + i * 0.1) * np.ones([778, 3])
          mesh.triangles = o3d.utility.Vector3iVector(F)  
          vis.add_geometry(mesh)
          meshes.append((mesh, mesh_t))

        #beta = tf.zeros([5, 10])
        pose = tf.zeros([1,16,3])
        frame_count = 0

        globalRunning = True
        while globalRunning:
            
            # Now we want to generate some random blend shapes. Let's start with no pertubations at all.
            if (frame_count >= 30):
                frame_count = 0
                pose = tf.random.normal([1, 16, 3])

                #beta = tf.random.normal([5, 10])
            
            # Generate the mesh by applying pertubations due to beta params.
            T_bar_batched = tf.repeat(tf.expand_dims(T_bar, axis=0), repeats=[5], axis=0)
            T_skinned, keypoints3D = lbs(pose, J, K, W, T_bar_batched)

            #T_posed_batched = T_bar_batched + blend_shape(beta, S)

            # Re-generate all hands from the batch, 
            # update the meshes, and update the renderer.
            for i in range(5):
              mesh, mesh_t = meshes[i]
              mesh.vertices = o3d.utility.Vector3dVector(T_skinned.numpy()[i,:,:] + mesh_t) 
              mesh.compute_vertex_normals()
              mesh.paint_uniform_color([0.75, 0.75, 0.75])
              vis.update_geometry(mesh)
            
            vis.poll_events()
            vis.update_renderer()

            time.sleep(1/60) # 1 s = 1000ms

            frame_count += 1

        vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

'''
NOTE(Noah): Note that the application will not close unless you force-quit/terminate it, i.e., the app
windows does not respond to clicking the close button.

-- Mouse view control --
  Left button + drag         : Rotate.
  Ctrl + left button + drag  : Translate.
  Wheel button + drag        : Translate.
  Shift + left button + drag : Roll.
  Wheel                      : Zoom in/out.

-- Keyboard view control --
  [/]          : Increase/decrease field of view.
  R            : Reset view point.
  Ctrl/Cmd + C : Copy current view status into the clipboard.
  Ctrl/Cmd + V : Paste view status from clipboard.

-- General control --
  Q, Esc       : Exit window.
  H            : Print help message.
  P, PrtScn    : Take a screen capture.
  D            : Take a depth capture.
  O            : Take a capture of current rendering settings.
'''
