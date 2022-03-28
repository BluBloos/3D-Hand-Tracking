import tensorflow as tf
from rodrigues import rodrigues

def blend_pose(pose, P):
    batch_size = pose.shape[0]
    I = tf.eye(3)
    rotationMatrix = tf.reshape(rodrigues(tf.reshape(pose,(-1, 3))),(batch_size, -1, 3, 3))
    pose_sum = tf.reshape(rotationMatrix[:, 1:, :, :]-I, (batch_size, -1))
    blend_pose = tf.reshape(tf.linalg.matmul(pose_sum, tf.reshape(P, (135, -1))), (batch_size, -1, 3))
    return blend_pose