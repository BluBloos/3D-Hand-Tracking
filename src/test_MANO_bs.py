# Authored by Noah Cabral
# This file contains a unit test for the formulation of B_s (the pertubations)
# to the MANO template mesh as per the Beta parameters.


import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
import os
import pickle
import open3d as o3d
import time 

def blend_shape(beta, S):
  # we note that l is the number of beta parameters.
  # v is the number of vertices in the MANO hand (778)
  # t is the dimensionality of each vertex (3).
  # b is the batch_size.
  return tf.einsum('bl,vtl->bvt', beta, S)

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

        globalRunning = True
        def key_callback(vis):
          global globalRunning
          globalRunning = False
          return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback( ord('.') , key_callback)
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

        beta = tf.zeros([5, 10])
        frame_count = 0

        
        while globalRunning:
            # Now we want to generate some random blend shapes. Let's start with no pertubations at all.
            if (frame_count >= 30):
                frame_count = 0
                beta = tf.random.normal([5, 10])
            
            # Generate the mesh by applying pertubations due to beta params.
            T_bar_batched = tf.repeat(tf.expand_dims(T_bar, axis=0), repeats=[5], axis=0)
            T_posed_batched = T_bar_batched + blend_shape(beta, S)

            # Re-generate all hands from the batch, 
            # update the meshes, and update the renderer.
            for i in range(5):
              mesh, mesh_t = meshes[i]
              mesh.vertices = o3d.utility.Vector3dVector(T_posed_batched.numpy()[i,:,:] + mesh_t) 
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
  .       : Exit window.
'''
