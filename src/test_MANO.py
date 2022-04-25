import tensorflow as tf
import numpy as np
import pickle
import time
from mano_layer import MANO_Model
from qmindcolors import cstr
import sys, os
import open3d as o3d

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
DEBUG = True

GRAYSCALE = False
IMAGE_SIZE = 224
BATCH_SIZE = 32
import imageio
import cv2 # opencv, for image resizing.
def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)
def resize(img, size):
    return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
def download_image(path):
  image = imageio.imread(path)
  _image = image.astype('float32')
  if GRAYSCALE:
      _image = rgb2gray(_image / 255)
  else:
      _image = _image / 255
  _image = resize(_image, IMAGE_SIZE)
  return _image

y_train = np.zeros( (32, 21, 3) )
k_train = np.zeros( (32, 3, 3) )
anno_train_path = '../RHD_small/training/anno_training.pickle'

# Remember, this is a dense load.
def load_anno(path, arr):
  anno_all = []
  count = 0
  with open(path, 'rb') as f:
    anno_all = pickle.load(f)

  for key, value in anno_all.items():
    if(count > BATCH_SIZE - 1):
      break
    kp_visible = (value['uv_vis'][:, 2] == 1)
    matrixK = value['K']
    k_train[count, :, :] = matrixK
    case1 = np.sum(kp_visible[0:21])
    case2 = np.sum(kp_visible[21:])
    if(case1 == 0):
      arr[count, :, :]= value['xyz'][21:42]
    else:
      arr[count, :, :]= value['xyz'][:21]
    count+=1

def demo2(render_RHD=False, offset=0):
    
    load_anno(anno_train_path, y_train)

    if DEBUG:
        enablePrint()
    else:
        blockPrint()

    mano_dir = os.path.join("..", "mano_v1_2")
    mpi_model = MANO_Model(mano_dir)    
    batch_size = 1
    beta = tf.zeros([batch_size, 10])
    pose = tf.repeat(tf.constant([[
        [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0],
        [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
        [0,0,0], [0,0,0], [0,0,0], [0,0,0]
    ]], dtype=tf.float32), repeats=[batch_size], axis=0)

    T_posed, keypoints3D = mpi_model(beta, pose, tf.zeros([batch_size, 3]))

    globalRunning = True
    def key_callback(vis):
        global globalRunning
        globalRunning = False
        return False

    # do open3d things.
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback( ord('.') , key_callback)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    vis.add_geometry(mesh_frame)
    
    gcs_path = '../SH_RHD'
    train_list = os.listdir(os.path.join(gcs_path, "training/color"))
    train_list.sort()
    
    if (render_RHD):
        for j in range(1):

            y_index = int(train_list[j + offset][0:5])
            print(cstr("y_index"), y_index)
            train_image_y = y_train[y_index] # [21, 3]
            k_y = k_train[y_index]   
            k_y_batched = np.repeat(np.expand_dims(k_y, axis=0), 21, axis=0 )
        
            # Put the hand in the center (but only subtract in the x: dimension).
            train_image_y -= np.array([0.0, 0.0, train_image_y[0][2]], dtype=np.float32)
            train_image_y *= 10

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
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(train_image_y)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)
            vis.update_geometry(line_set)

            i = 0
            for i in range(21):
                keypoint = train_image_y[i]
                #print(cstr("squeezed"), keypoint.numpy())
                msphere = o3d.geometry.TriangleMesh.create_sphere(0.05)
                msphere.paint_uniform_color([0.75, 1 - (j+1) / 5, (j+1) / 5])
                msphere.compute_vertex_normals()
                msphere.translate(keypoint)
                vis.add_geometry(msphere)
                vis.update_geometry(msphere)     
                #render.scene.add_geometry("sphere{}".format(i), msphere, yellow)
    
    # [bs, 16, 3]
    keypoints3D_pylist = tf.unstack( keypoints3D, axis=1 )
    
    i = 0
    for keypoint in keypoints3D_pylist:
        keypoint = tf.squeeze(keypoint)
        msphere = o3d.geometry.TriangleMesh.create_sphere(0.05)
        msphere.paint_uniform_color([0, 0.75, 0])
        msphere.compute_vertex_normals()
        msphere.translate(keypoint.numpy() * 10)
        vis.add_geometry(msphere)
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
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(tf.squeeze(keypoints3D).numpy() * 10)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)
    vis.update_geometry(line_set)

    # add the MANO mesh as well.
    T_posed_scaled = T_posed * 10
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(mpi_model.F.numpy())
    mesh.vertices = o3d.utility.Vector3dVector(T_posed_scaled[0, :, :].numpy()) 
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    vis.add_geometry(pcd)   
    vis.update_geometry(pcd)  
    
    while globalRunning:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1/60) # 1 s = 1000ms

    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

if __name__ == "__main__":
    demo2(render_RHD=True, offset=1)