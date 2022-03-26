from rodrigues import rodrigues
import tensorflow as tf
import numpy as np
import cv2
from keras import Model
import os
import pickle
import open3d as o3d
import time 

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

        T_bar = tf.convert_to_tensor(manoRight['v_template'], dtype=tf.float32) # shape of (778, 3)
        F = np.array(manoRight['f'], dtype=np.int32)  

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(mesh)
        vis.add_geometry(mesh_frame)
        angleAxis = tf.constant([[np.pi/20, 0, 0]])
        rMatrix = rodrigues(angleAxis)
        frame_count = 0

        globalRunning = True
        while globalRunning:
            # Now we want to generate some random blend shapes. Let's start with no pertubations at all.
            if (frame_count >= 90):
                frame_count = 0
                angleAxis = tf.random.uniform([1,3], minval = -2*np.pi, maxval = 2*np.pi)
                angleAxis /= tf.norm(angleAxis, axis = 1, keepdims = True)
                angleAxis *= np.pi/20
                rMatrix = rodrigues(angleAxis)
            
            # Generate the mesh by applying pertubations due to beta params.
            T_bar = tf.linalg.matmul(T_bar, rMatrix)

            # Re-generate the hand, update mesh, and update renderer. 
            mesh.vertices = o3d.utility.Vector3dVector(T_bar.numpy()[0,:,:])
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.75, 0.75, 0.75])
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            time.sleep(1/60) # 1 s = 1000ms

            frame_count += 1

        vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
