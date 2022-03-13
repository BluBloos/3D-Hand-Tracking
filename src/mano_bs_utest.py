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

def blend_shape(beta, S):
  # we note that l is the number of beta parameters.
  # v is the number of vertices in the MANO hand (778)
  # t is the dimensionality of each vertex (3).
  # b is the batch_size.
  return tf.einsum('bl,vtl->bvt', [beta, S])

try:
    # Here we load in the learned MANO parameters.
    mano_dir = "../mano_v1_2/models"
    file_path = os.path.join(mano_dir, "models", "MANO_RIGHT.pkl")
    manoRight = pickle.load(open(file_path, 'rb'), encoding='latin1')
    S = np.array(manoRight['shapedirs'], dtype=np.float32) # shape of S is (778, 3, 10)
    T_bar = np.array(manoRight['v_template'], dtype=np.float32) # shape of (778, 3)
    F = np.array(manoRight['f'], dtype=np.int32)  

    # Now we want to generate some random blend shapes. Let's start with no pertubations at all.
    beta = np.zeros([1, 10])

    # Generate the mesh by applying pertubations due to beta params.
    T_posed = T_bar + blend_shape(beta, S)

    ########################################
    ### Quick visualization using Open3D ###
    ########################################
    # Create a reference frame 10 cm
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # Draw mesh model
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(T_posed[0,:,:])
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.75, 0.75, 0.75])

    o3d.visualization.draw_geometries([mesh, mesh_frame])
    #o3d.visualization.draw_geometries([ls, mesh_frame] + mesh_spheres)

except:
    print("Oops! Something went wrong.\n\
    It's likely that MANO_RIGHT.pkl was not found.\n\
    Check that this file exists in mano_v1_2/models")