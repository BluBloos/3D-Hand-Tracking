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
        mano_dir = "../mano_v1_2"
        file_path = os.path.join(mano_dir, "models", "MANO_RIGHT.pkl")
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

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(mesh)
        vis.add_geometry(mesh_frame)

        beta = tf.zeros([1, 10])
        frame_count = 0

        globalRunning = True
        while globalRunning:
            # Now we want to generate some random blend shapes. Let's start with no pertubations at all.
            if (frame_count >= 30):
                frame_count = 0
                beta = tf.random.normal([1, 10])
            
            # Generate the mesh by applying pertubations due to beta params.
            T_posed = T_bar + blend_shape(beta, S)

            # Re-generate the hand, update mesh, and update renderer. 
            mesh.vertices = o3d.utility.Vector3dVector(T_posed.numpy()[0,:,:])
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
