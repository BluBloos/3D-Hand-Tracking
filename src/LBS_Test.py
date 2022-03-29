# Authored by Lucas Coster
# Test for the MANO Linear Blend Skinning functions

from fnmatch import translate
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
import pickle
import time

from mano_bs import blend_shape
from mano_bp import blend_pose
from mano_bp import rmatrix_from_pose
from rodrigues import rodrigues

def vertices2joints(vert, J_):
    return tf.einsum('bvt,jv->bjt', vert, J_)

def transform_matrix(R, t):
    bs = R.shape[0]
    T = tf.concat([R,t], axis=2) # 3x4 (3 rows, 4 columns)
    row = tf.constant([[0,0,0,1]], dtype = tf.float32)
    row_batched = tf.repeat(tf.expand_dims(row, axis=0), repeats=[bs], axis=0)
    T = tf.concat([T,row_batched], axis = 1)
    return T

def lbs(beta, pose, J, K, W, S, P, T_bar):

    bs = pose.shape[0]
    T_shaped = T_bar + blend_shape(beta, S)
    J_rest = vertices2joints(T_shaped, J)
    T_posed = T_shaped + blend_pose(pose, P)
    rmats = rmatrix_from_pose(pose)

    ### BATCH_RIGID_TRANFORM ###
    # j_transformed, A = batch_rigid_transform(rmatrix, j_rest, K_)
    joints = tf.expand_dims(J_rest, axis =-1)
    parents = K
    rel_joints = tf.identity(joints)

    # TODO: Investigate if this is the right way to handle the root joint...
    parents -= tf.concat([ tf.constant([4294967295], dtype=tf.int64), tf.zeros( (15), dtype=tf.int64 ) ], axis=0)
    
    rel_joints -= tf.gather(joints[:, :], indices=parents, axis=1)
    
    print("rel_joints=\n", rel_joints)
    print("rmats=\n", rmats)
    tm = transform_matrix(tf.reshape(rmats, (-1, 3, 3)), tf.reshape(rel_joints, (-1, 3, 1)))
    print("tm=\n", tm)

    # We note that tm gets collapsed along the batch dimension (combining batches with 2nd dim)
    


    ### BATCH_RIGID_TRANFORM ###


    '''
    W_bar_batched = tf.repeat(tf.expand_dims(W_, axis=0), repeats=[bs], axis=0) # TODO: This might be totally false.                     
    T = tf.matmul(W_bar_batched, tf.reshape(A,(bs, -1, 16)))
    T = tf.reshape(T, (bs, -1, 4, 4))

    ones = tf.ones([bs, v_posed.shape[1],1], dtype=tf.float32)
    v_posed_homo = tf.concat([v_posed, ones], axis=2)
    v_homo = tf.matmul(T , tf.expand_dims(v_posed_homo, axis = -1))

    vertices = v_homo[:, :, :3, 0]
    '''

    # return vertices, j_transformed


def batch_rigid_transform(rmats, joints, parents):
    # transforms_matrix = tf.reshape(transforms_matrix, ((-1, joints.shape[1], 4, 4)))                     
    # transform_chain = [transforms_matrix[:,0]]

    '''
    for i in range(1, parents.shape[0]):
        # subtract joint location at rest pose
        curr_res = tf.matmul(transform_chain[parents[i]],transforms_matrix[:,i])
        transform_chain.append(curr_res)

    transforms = tf.stack(transform_chain, axis=1)

    # the last coloumn of the transfer contains the posed joints
    posed_joints = transforms[:,:,3,3]

    # TODO: We want to change this from F. Definitely cannot be this.
    joints_homogen = tf.pad(joints,[0,0,0,1])
    rel_transforms = transforms - tf.pad(tf.matmul(transforms, joints_homogen), [3,0,0,0,0,0,0,0])
    '''
    pass

import open3d as o3d

def demo():
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
        K = tf.convert_to_tensor(manoRight['kintree_table'][0], dtype=tf.int64)
        # Joint regressor that are learned (J), Shape=(16,778), type=float64   
        J = tf.convert_to_tensor(manoRight['J_regressor'].todense(), dtype=tf.float32)
        # Weights that are learned (W), Shape=(778,16), type=float64       
        W = tf.convert_to_tensor(manoRight['weights'], dtype=tf.float32)             
        
        
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
            #T_bar_batched = tf.repeat(tf.expand_dims(T_bar, axis=0), repeats=[5], axis=0)
            T_skinned, keypoints3D = lbs(pose, J, K, W, T_bar)

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


def unit_test():

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
        P = tf.convert_to_tensor(manoRight['posedirs'] , dtype=tf.float32)

        # Kinematic tree defining the parent joint (K), Shape=(16,), type=int64
        K = tf.convert_to_tensor(manoRight['kintree_table'][0], dtype=tf.int64)
        
        # Joint regressor that are learned (J), Shape=(16,778), type=float64   
        J = tf.convert_to_tensor(manoRight['J_regressor'].todense(), dtype=tf.float32)
        
        # Weights that are learned (W), Shape=(778,16), type=float64       
        W = tf.convert_to_tensor(manoRight['weights'], dtype=tf.float32)


        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        batch_size = 1
        beta = tf.zeros([batch_size, 10])
        pose = tf.zeros([batch_size, 16, 3])
        T_bar_batched = tf.repeat(tf.expand_dims(T_bar, axis=0), repeats=[batch_size], axis=0)
        
        lbs(beta, pose, J, K, W, S, P, T_bar_batched)
        
        '''
        j_rest = vertices2joints(T_bar_batched, J)
        # whenever we do a -1 in a reshape, this set the dimension size to whatever it needs to be
        # so that we preserve the size before reshaping.
        rmatrix = tf.reshape(rodrigues(tf.reshape(pose, (-1,3))), (bs,-1,3,3)) 
        print("rmatrix", rmatrix)
        batch_rigid_transform(rmatrix, j_rest, K)

        # create the meshes
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(mesh_frame)  
        
        for i in range(16):
          mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
          mesh_sphere.compute_vertex_normals()
          mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
          print("j_rest[0, i].numpy()", j_rest[0, i].numpy())
          mesh_sphere.translate( j_rest[0, i].numpy() )
          vis.update_geometry(mesh_sphere)
        
        globalRunning = True
        while globalRunning:
            vis.poll_events()
            vis.update_renderer()
        '''

if __name__ == "__main__":

    #demo()

    unit_test()
    
    

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
