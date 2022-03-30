'''

NOTE: For the better estimation of a MANO mesh we must render mesh to a depth image
-> weak supervision via RHD ground truth depth images.

^ this is an after CUCAI task.

Step 1) Write a visualization for rendering the 3D keypoints of a MANO hand.
Step 2) Then use this to verfiy the pseudo LBS func.
Step 3) Change lbs accordingly to get things to a working state.
'''

# Authored by Lucas Coster, Noah Cabral, Max Vincent
# Test for the MANO Linear Blend Skinning functions

from fnmatch import translate
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import pickle
import time

from mano_bs import blend_shape
from mano_bp import blend_pose
from mano_bp import rmatrix_from_pose
from rodrigues import rodrigues

import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
DEBUG = True
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def cstr(str): # cyan string
    return bcolors.OKCYAN + str + bcolors.ENDC

def vertices2joints(vert, J_):
    return tf.einsum('bvt,jv->bjt', vert, J_)

def transform_matrix(R, t):
    bs = R.shape[0]
    T = tf.concat([R,t], axis=2) # 3x4 (3 rows, 4 columns)
    row = tf.constant([[0,0,0,1]], dtype = tf.float32)
    row_batched = tf.repeat(tf.expand_dims(row, axis=0), repeats=[bs], axis=0)
    T = tf.concat([T,row_batched], axis = 1)
    return T

# currently a pseudo lbs.
def lbs(beta, pose, J, K, W, S, P, T_bar):

    bs = pose.shape[0]
    T_shaped = T_bar + blend_shape(beta, S)
    J_rest = vertices2joints(T_shaped, J)
    T_posed = T_shaped + blend_pose(pose, P)
    rmats = rmatrix_from_pose(pose)
    #rmats = tf.eye(3, num_columns=3, batch_shape=[bs, 16], dtype=tf.float32)

    ### BATCH_RIGID_TRANFORM ###
    # j_transformed, A = batch_rigid_transform(rmatrix, j_rest, K_)
    print(cstr("J_rest=\n"), J_rest)
    joints = tf.expand_dims(J_rest, axis =-1) # converts J_rest into a batch of column vectors.
    print(cstr("joints=\n"), joints)
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
    print(cstr("rel_joints=\n"), rel_joints)

    print(cstr("rmats=\n"), rmats)
    tm = transform_matrix(tf.reshape(rmats, (-1, 3, 3)), tf.reshape(rel_joints, (-1, 3, 1)))
    print(cstr("tm=\n"), tm)

    # We note that tm gets collapsed along the batch dimension (combining batches with 2nd dim)
    # So we transform it back.
    tm = tf.reshape(tm, ((-1, joints.shape[1], 4, 4)))
    print(cstr("tm(after reintroducing batches)=\n"), tm)

    
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

    print(cstr("Gk_pylist=\n"), Gk_pylist)
    Gk = tf.stack(Gk_pylist, axis=1)
    print(cstr("Gk=\n"), Gk)

    # Why is it that these are actually the posed_joints?
    # because the joints are already in their own ref frame.
    posed_joints = tf.slice( Gk[:, :, :, 3], [0, 0, 0], [bs, 16, 3])
    # posed_joints = Gk[:, :, :, 3]
    print(cstr("posed_joints"), posed_joints)

    return posed_joints

    '''
    # the last coloumn of the transfer contains the posed joints
    posed_joints = transforms[:,:,3,3]

    # TODO: We want to change this from F. Definitely cannot be this.
    joints_homogen = tf.pad(joints,[0,0,0,1])
    rel_transforms = transforms - tf.pad(tf.matmul(transforms, joints_homogen), [3,0,0,0,0,0,0,0])
    return posed_joints, rel_transforms
    '''

    ### BATCH_RIGID_TRANFORM ###

    '''
    # take G_k' and weight it by W.
    W_bar_batched = tf.repeat(tf.expand_dims(W_, axis=0), repeats=[bs], axis=0) # TODO: This might be totally false.                     
    T = tf.matmul(W_bar_batched, tf.reshape(A, (bs, -1, 16)))
    T = tf.reshape(T, (bs, -1, 4, 4))

    # run the points through T, which is the G_k' but weighted by W.
    ones = tf.ones([bs, T_posed.shape[1],1], dtype=tf.float32)
    T_posed_homo = tf.concat([T_posed, ones], axis=2)
    T_homo = tf.matmul(T , tf.expand_dims(T_posed_homo, axis = -1))
    vertices = T_homo[:, :, :3, 0]
    '''

    # return vertices, j_transformed

import open3d as o3d
import open3d.visualization.rendering as rendering

# The goal of the demo here is to use open3d to render something transparently onto a backdrop.
def demo():
    
    if DEBUG:
        enablePrint()
    else:
        blockPrint()

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
        
        keypoints3D = lbs(beta, pose, J, K, W, S, P, T_bar_batched) # [bs, 16, 3]

        # render the 3d keypoints and display the image.
        render = rendering.OffscreenRenderer(1920, 1080)
        yellow = rendering.MaterialRecord()
        yellow.base_color = [1.0, 0.75, 0.0, 1.0]
        yellow.shader = "defaultLit"

        green = rendering.MaterialRecord()
        green.base_color = [0.0, 0.5, 0.0, 0.5] # [r,g,b,a]
        green.shader = "defaultLit"
        green.has_alpha = True

        grey = rendering.MaterialRecord()
        grey.base_color = [0.7, 0.7, 0.7, 1.0]
        grey.shader = "defaultLit"

        white = rendering.MaterialRecord()
        white.base_color = [1.0, 1.0, 1.0, 1.0]
        white.shader = "defaultLit"

        cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
        cyl.compute_vertex_normals()
        cyl.translate([-2, 0, 1.5])
        sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
        sphere.compute_vertex_normals()
        sphere.translate([-2, 0, 3])

        box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
        box.compute_vertex_normals()
        box.translate([-1, -1, 0])
        solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
        solid.compute_triangle_normals()
        solid.compute_vertex_normals()
        solid.translate([0, 0, 1.75])

        render.scene.add_geometry("cyl", cyl, green)
        #render.scene.add_geometry("sphere", sphere, yellow)

        # [bs, 16, 3]
        keypoints3D_pylist = tf.unstack( keypoints3D, axis=1 )
        print(cstr("keypoints3D_pylist"), keypoints3D_pylist)

        i = 0
        for keypoint in keypoints3D_pylist:
            keypoint = tf.squeeze(keypoint)
            print(cstr("squeezed"), keypoint.numpy())
            msphere = o3d.geometry.TriangleMesh.create_sphere(0.05)
            msphere.compute_vertex_normals()
            msphere.translate(keypoint.numpy() * 10)
            render.scene.add_geometry("sphere{}".format(i), msphere, yellow)
            i += 1

        # add the MANO mesh as well.
        T_bar_scaled = T_bar * 10
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh.vertices = o3d.utility.Vector3dVector(T_bar_scaled) 
        mesh.compute_vertex_normals()

        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        render.scene.add_geometry("pcd", pcd, green)

        #render.scene.add_geometry("mesh", mesh, green)
        
        render.setup_camera(60.0, [0, 0, 0], [2.5, 2.5, 2.5], [0, 0, 1])
        render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                        75000)
        render.scene.scene.enable_sun_light(True)
        render.scene.show_axes(True)

        img = render.render_to_image()
        print("Saving image at test.png")
        o3d.io.write_image("test.png", img, 9)


def unit_test():

    if DEBUG:
        enablePrint()
    else:
        blockPrint()

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

if __name__ == "__main__":

    demo()

    #unit_test()
    
    

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
