# Author(s): Oscar Lu, Noah Cabral, Max Vincent, Lucas Coster

import tensorflow as tf
from mobilehand import ortho_camera
from mobilehand import camera_extrinsic

# TODO: Rename this to L2.
# works for [bs, point_count, 3]
def mse(pred, gt):
  bs = pred.shape[0]
  point_count = pred.shape[1]
  mse = tf.reduce_sum(tf.reduce_sum(tf.square(pred - gt), axis=2), axis=1) / point_count
  loss = tf.reduce_sum(mse, axis=0) / bs 
  return loss

#_mse = tf.keras.losses.MeanSquaredError()
_mse = mse

# pred is a tensor with shape of [bs, 21, 3]. These are the estimated 3D keypoints by MANO.
# gt is a tensor with shape of [bs, 21, 3]. These are the ground-truth 3D keypoints provided by RHD.
def LOSS_2D(scale, pred, gt):
  pred = ortho_camera(camera_extrinsic(scale, pred))
  gt = ortho_camera(gt)
  return _mse(pred, gt)

# no scale dependence -> everything is normalized.
# gt must be scaled down.
def LOSS_3D(pred, gt_prime):
  # MSE = np.square(np.subtract(pred,gt)).mean()
  # pred = _camera_extrinsic_post_rot(depth, scale, pred)
  return _mse(pred, gt_prime)

# takes in scale w/ shape = [bs, 1, 1]
def LOSS_CAM(scale, gt_scale):
  # Applying L2 loss with batch sum reduction.
  return  _mse(scale, gt_scale)

# beta is a tensor with shape of [bs, 10]. These are the estimated beta parameters that get fed to MANO.
# pose is a tensor with shape of [bs, 48]. These are the estimated pose parameters that get fed to MANO.
def LOSS_REG(beta, pose, L, U):
  # U and L are upper and lower limits for the alpha params (which are the things that get mapped into theta MANO params).
  # U and L are only shape R^45.
  pose = pose[ :, 3:]
  bs = pose.shape[0]
  loss = tf.reduce_sum(tf.square(beta), axis=1)
  loss += tf.reduce_sum(
    tf.math.maximum(L - pose, tf.zeros(pose.shape)) + \
    tf.math.maximum(pose - U, tf.zeros(pose.shape)), axis=1
  )
  loss = tf.reduce_sum(loss, axis=0) / bs
  return loss

# Master loss function
def LOSS(beta, pose, L, U, scale, pred, gt, gt_scale):
  gt_prime = gt / gt_scale # Inverse scale transform.
  return 1e2 * (LOSS_2D(scale, pred, gt) + LOSS_3D(pred, gt_prime) + \
    LOSS_CAM(scale, gt_scale))
  

def distance(arr1, arr2):
  diff = tf.math.subtract(arr1, arr2)
  distance = tf.norm(diff, axis = 2)
  distance = tf.squeeze(distance)
  return distance
