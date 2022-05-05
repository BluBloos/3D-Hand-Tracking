import tensorflow as tf
import numpy as np
import pickle
import time
from mano_layer import MANO_Model
from qmindcolors import cstr
import sys, os
import open3d as o3d
import socket
import sys
from _thread import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
DEBUG = True

########################################### MODEL/ANNOT/DATA LOADING ###########################################
anno_train_path = '../RHD_small/training/anno_training.pickle'
anno_eval_path = '../RHD_small/evaluation/anno_evaluation.pickle'
from anno_load import load_anno_all
from anno_load import y_train
from anno_load import y_test
from data_load import IMAGE_SIZE, download_image
from data_load import IMAGE_CHANNELS
from mobilehand import camera_extrinsic
BATCH_SIZE = 32
MANO_DIR = os.path.join("..", "mano_v1_2")
from mobilehand import MAKE_MOBILE_HAND
model = MAKE_MOBILE_HAND(IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, MANO_DIR)
gcs_path = '../SH_RHD'
train_list = os.listdir(os.path.join(gcs_path, "training/color"))
eval_list = os.listdir(os.path.join(gcs_path, "evaluation/color"))
train_list.sort()
eval_list.sort()
########################################### MODEL/ANNOT/DATA LOADING ###########################################

# given an open3D visualizer, we want to setup the scene.
def update_scene(vis, img_in, annot_3D):
    
    global line_set
    global line_set_spheres
    global mano_spheres
    global mano_line_set
    global pcd

    try:
        # k_y = k_train[y_index]
        # k_y_batched = np.repeat(np.expand_dims(k_y, axis=0), 21, axis=0 )
        # Put the hand in the center (but only subtract in the xy dimensions). Keep the depth.
        annot_3D -= np.array([annot_3D[0][0], annot_3D[0][1], 0.0], dtype=np.float32)
        mpi_model = MANO_Model(MANO_DIR) 

        # build the lines for keypoint annotations
        lines = [ ]
        for i in range(1, 21):
            lines.append( 
                [
                    i, 
                    mpi_model.RHD_K.numpy()[i]
                ]
            )
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set.points = o3d.utility.Vector3dVector(annot_3D)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(line_set)

        i = 0
        for i in range(21):
            keypoint = annot_3D[i]
            msphere = line_set_spheres[i]
            msphere.paint_uniform_color([0.75, 1 - (0+1) / 5, (0+1) / 5])
            msphere.compute_vertex_normals()
            msphere.translate(-msphere.get_center()) # reset sphere
            msphere.translate(keypoint)
            vis.update_geometry(msphere)     
        
        scale = np.sqrt(np.sum(np.square(annot_3D[0] - annot_3D[8]), axis=0, keepdims=True)) / 0.1537328322252615 
        scale = np.repeat(np.expand_dims(scale, axis=0), repeats=32, axis=0)
        z_depth = tf.repeat(tf.constant([[0, 0, annot_3D[0][2]]]), repeats=32, axis=0)
        # Step 1 is to use the eval_image in a forward pass w/ the model to generate a ckpt_image.
        _beta, _pose, T_posed, _keypoints3D, cam_R = model(
            np.repeat(np.expand_dims(img_in, 0), 32, axis=0))
        
        T_posed = camera_extrinsic(cam_R, z_depth, scale, T_posed)
        keypoints3D = camera_extrinsic(cam_R, z_depth, scale, _keypoints3D)

        # need to consider just 1 of the 32 outputs (because things have a batch size)
        T_posed = T_posed[0]
        keypoints3D = keypoints3D[0]

        print(cstr("keypoints3D"), keypoints3D)
        
        keypoints3D_pylist = tf.unstack( keypoints3D, axis=0 )

        print(cstr("keypoints3D_pylist"), keypoints3D_pylist)
        
        i = 0
        for keypoint in keypoints3D_pylist:
            keypoint = tf.squeeze(keypoint)
            msphere = mano_spheres[i]
            msphere.paint_uniform_color([0, 0.75, 0])
            msphere.compute_vertex_normals()    
            msphere.translate(-msphere.get_center()) # reset sphere
            msphere.translate(keypoint.numpy())
            vis.update_geometry(msphere)     
            i += 1
        
        # build the lines
        lines = [ ]
        for i in range(1, 21):
            lines.append( 
                [
                    i, 
                    mpi_model.RHD_K.numpy()[i]
                ]
            )
        colors = [[1, 0, 0] for i in range(len(lines))]
        mano_line_set.points = o3d.utility.Vector3dVector(tf.squeeze(keypoints3D).numpy())
        mano_line_set.lines = o3d.utility.Vector2iVector(lines)
        mano_line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(mano_line_set)

        # add the MANO mesh as well.
        T_posed_scaled = T_posed
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(mpi_model.F.numpy())
        mesh.vertices = o3d.utility.Vector3dVector(T_posed_scaled.numpy()) 
        mesh.compute_vertex_normals()
        _pcd = mesh.sample_points_uniformly(number_of_points=1000)
        pcd.points = _pcd.points # lol hopefully this works?
        vis.update_geometry(pcd)

        # reposition the camera
        vc = vis.get_view_control()
        vc.set_lookat( annot_3D[0] )

    except Exception as e:
        print(e)

def clientthread(conn, addr, vis_lock): 
    global vis
    conn.send(b"Please enter commands!")
    while True:
        try:
            message = conn.recv(2048)
            if message:
                message_str = message.decode("utf-8")
                print(cstr("recieved message ="), message_str)
                if message_str[0] == "l":
                    # load command, we want to load a new image for inference
                    if message_str[1] == "t":
                        # want to load from the training set.
                        t_index = int(message_str[2:])
                        print(cstr("loading training image"), t_index)
                        annot = y_train[t_index]
                        t_img = download_image(gcs_path, "training", t_index)
                        vis_lock.acquire()
                        update_scene(vis, t_img, annot)
                        vis_lock.release()
                    elif message_str[1] == "e":
                        # want to load from the evaluation set.
                        e_index = int(message_str[2:])
                        print(cstr("loading evaluation image"), e_index)
                        annot = y_test[e_index]
                        e_img = download_image(gcs_path, "evaluation", e_index)
                        vis_lock.acquire()
                        update_scene(vis, e_img, annot)
                        vis_lock.release()
                    elif message_str[1] == "c":
                        # Want to load a ckpt!!
                        ckpt = int(message_str[2:])
                        checkpoint_path = os.path.join("..", "checkpoints/")
                        file_path = os.path.join(checkpoint_path, "cp-{:04d}.ckpt".format(ckpt))
                        model.load_weights(file_path)
                        print(cstr("loaded model weights:"), ckpt)
            else:
                remove(conn)
        except:
            continue

def remove(connection):
    if connection in list_of_clients:
        list_of_clients.remove(connection)

def server_thread(server, vis_lock):
    while True:
        conn, addr = server.accept()
        list_of_clients.append(conn)
        print(cstr(addr[0] + " connected"))
        start_new_thread(clientthread, (conn, addr, vis_lock))

# global variables
list_of_clients = []
vis = None
pcd = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()
mano_line_set = o3d.geometry.LineSet()
SPHERE_RADIUS = 0.005
line_set_spheres = [ o3d.geometry.TriangleMesh.create_sphere(SPHERE_RADIUS) for i in range(21) ]
mano_spheres = [ o3d.geometry.TriangleMesh.create_sphere(SPHERE_RADIUS) for i in range(21) ]
vis_lock = allocate_lock()
globalRunning = True

if __name__ == "__main__":

    load_anno_all(anno_train_path, anno_eval_path)
    if DEBUG:
        enablePrint()
    else:
        blockPrint()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # Add geometries to visualization
    vis.add_geometry(mesh_frame)
    vis.add_geometry(line_set)
    vis.add_geometry(mano_line_set)
    vis.add_geometry(pcd)
    for i in range(21):
        vis.add_geometry(line_set_spheres[i])
        vis.add_geometry(mano_spheres[i])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    addr = "127.0.0.1"
    port = 5000
    server.bind((addr, port))
    #listens for 100 active connections.
    server.listen(100)
    start_new_thread(server_thread, (server, vis_lock))

    while globalRunning:
        vis_lock.acquire()
        vis.poll_events()
        vis.update_renderer()
        vis_lock.release()
        time.sleep(1/60) # 1 s = 1000ms

    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)