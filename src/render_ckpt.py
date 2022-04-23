
import open3d as o3d
import open3d.visualization.rendering as rendering
from mano_layer import MANO_Model
import numpy as np
import os
import tensorflow as tf
from qmindcolors import cstr
import matplotlib.pyplot as plt

# with reference to this post https://www.codeitbro.com/send-email-using-python/#step-1-8211connect-to-the-mail-server. 
# Seems pretty bad tbh but it's gonna do the job??
import smtplib
import imghdr
from email.message import EmailMessage

Sender_Email = "acc.cnoah@gmail.com"
Reciever_Email = "cnoah1705@gmail.com"
Password = "htqkbbitakdonazr"

def send_email(image_file):
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "New Checkpoint" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('Image attached!') 

    with open(image_file, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)              
        smtp.send_message(newMessage)

# ckpt_index is an index for the current checkpoint that the model param is loaded
# with weights.  
def render_checkpoint_image(ckpt_path, ckpt_index, model, eval_image, annot, template_override=False):

    # Step 1 is to use the eval_image in a forward pass w/ the model to generate a chkpt_image.
    _beta, _pose, T_posed, keypoints3D = model(np.repeat(np.expand_dims(eval_image, 0), 32, axis=0))

    render = rendering.OffscreenRenderer(1080, 1080)
    
    # TODO(Noah): Reloading MANO here is sort of redundant. We should expose the MANO params on the
    # model or something like that.
    mano_dir = os.path.join("..", "mano_v1_2")
    mpi_model = MANO_Model(mano_dir)  

    if template_override:
        #print(cstr("template_override!"))
        batch_size = 1
        beta = tf.zeros([batch_size, 10])
        pose = tf.repeat(tf.constant([[
            [1.57/2,0,0], # Root
            [0,1.57/2,0], # 
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]], dtype=tf.float32), repeats=[batch_size], axis=0)

        _beta, _pose, T_posed, keypoints3D = mpi_model(beta, pose, 
            tf.constant([[0,0,0]]))

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
    line_set.points = o3d.utility.Vector3dVector(annot)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    render.scene.add_geometry("line_set_RHD", line_set, red)
    i = 0
    for i in range(21):
        keypoint = annot[i]
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
    
    center = [0,0, annot[0][2]] # select the root annotation location
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
    #o3d.io.write_image(img_filepath, img, 9)

    # TODO(Noah): Make it so that we render all of these images onto one big image in a grid
    # for ease of viewing. AND, we can include the original image (the one passed into the model) 
    # into the grid.
    # below, we are going to grab TWO more viewpoints of the same image
    #img_filepath = os.path.join(checkpoint_path, "image-left-{:0d}.png".format(ckpt_index))
    render.setup_camera(60.0, center, [center[0]-z_dist * 0.7071, center[1], center[2]-z_dist * 0.7071], [0, -1, 0])  
    img2 = render.render_to_image()
    #o3d.io.write_image(img_filepath, img, 9)
    #img_filepath = os.path.join(checkpoint_path, "image-right-{:0d}.png".format(ckpt_index))
    render.setup_camera(60.0, center, [center[0]+z_dist * 0.7071, center[1], center[2]-z_dist * 0.7071], [0, -1, 0])  
    img3 = render.render_to_image()  
    #o3d.io.write_image(img_filepath, img, 9)
    
    w = 10
    h = 10
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(eval_image)
    #fig.add_subplot(rows, columns, 2)
    #plt.imshow(img2)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img1)
    #fig.add_subplot(rows, columns, 4)
    #plt.imshow(img3)

    plt.savefig(img_filepath)


    #send_email(img_filepath)
    #print(cstr("Sent email of"), img_filepath)