import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import os
import tensorflow as tf
from qmind_lib import cstr
import matplotlib.pyplot as plt
from mobilehand import camera_extrinsic

# ckpt_index is an index for the current checkpoint that the model param is loaded
# with weights.
def render_checkpoint_image(ckpt_path, ckpt_index, model, eval_image, annot):

    annot_2D, annot_3D, annot_K = annot

    # Step 1 is to use the eval_image in a forward pass w/ the model to generate a ckpt_image.   
    _beta, _pose, T_posed, _keypoints3D, scale = model(
        np.repeat(np.expand_dims(eval_image, 0), 32, axis=0))

    T_posed = camera_extrinsic(scale, T_posed)
    keypoints3D = camera_extrinsic(scale, _keypoints3D)

    render = rendering.OffscreenRenderer(1080, 1080)
    
    mpi_model = model.mano_model

    green = rendering.MaterialRecord()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"
    red = rendering.MaterialRecord()
    red.base_color = [0.5, 0.0, 0.0, 1.0]
    red.shader = "defaultLit"  
    yellow = rendering.MaterialRecord()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"  

    # k_y_batched = np.repeat(np.expand_dims(annot, axis=0), 21, axis=0)
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
    line_set.points = o3d.utility.Vector3dVector(annot_3D)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    render.scene.add_geometry("line_set_RHD", line_set, red)
    i = 0
    for i in range(21):
        keypoint = annot_3D[i]
        msphere = o3d.geometry.TriangleMesh.create_sphere(0.005)
        # msphere.paint_uniform_color([0.75, 1 - (j+1) / 5, (j+1) / 5])
        msphere.compute_vertex_normals()
        msphere.translate(keypoint)
        render.scene.add_geometry("sphere_RHD{}".format(i), msphere, yellow)

    # [bs, 16, 3]
    keypoints3D_pylist = tf.unstack( keypoints3D, axis=1 )
    i = 0
    for keypoint in keypoints3D_pylist:
        keypoint = keypoint[0, :]
        msphere = o3d.geometry.TriangleMesh.create_sphere(0.005)
        msphere.paint_uniform_color([0, 0.75, 0])
        msphere.compute_vertex_normals()
        msphere.translate(keypoint.numpy())   
        render.scene.add_geometry("sphere{}".format(i), msphere, green)
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
    line_set.points = o3d.utility.Vector3dVector(keypoints3D[0, :, :].numpy())
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    render.scene.add_geometry("line_set", line_set, red)

    # add the MANO mesh as well.
    T_posed_scaled = T_posed
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(mpi_model.F.numpy())
    mesh.vertices = o3d.utility.Vector3dVector(T_posed_scaled[0, :, :].numpy()) 
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    render.scene.add_geometry("pcd", pcd, red)  
    
    center = [0,0, annot_3D[0][2]] # Select the root annotation location
    z_dist = 0.5  

    # if we want to understand the params of the setup_camera func, it goes like this.
    # (FOV, center, eye, up)
    # when we setup the camera like we are doing here, it's the same as the openGL gl.lookAt()
    # So for the explanation of the center, eye, and up params, see here 
    # https://stackoverflow.com/questions/21830340/understanding-glmlookat  
    render.setup_camera(60.0, center, [center[0], center[1], center[2]-z_dist], [0, -1, 0])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                    75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(False)

    img1 = render.render_to_image()
    img_filepath = os.path.join(ckpt_path, "image-{:0d}.png".format(ckpt_index))
    print(cstr("Saving image(s) at"), img_filepath)
    
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 1

    #print(eval_image)

    fig.add_subplot(rows, columns, 1)
    plt.imshow( eval_image.astype(np.uint8) )
    fig.add_subplot(rows, columns, 2)
    plt.imshow( np.array(img1).astype(np.uint8) )
    plt.savefig(img_filepath)
    plt.close()