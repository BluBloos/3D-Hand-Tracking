import tensorflow as tf
from tensorflow.keras import Model

from mobilenet import MobileNetV3Small
from ireg import IterativeRegression
from mano import MANO_Model

# take 3D points and apply orthographic projection.
def ortho_camera(points):
  intrinsic = tf.constant(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0]
    ], dtype=tf.float32)
  return tf.matmul(points, intrinsic)

# take 3D points and apply the extrinsic camera transform.
def camera_extrinsic(scale, points):
  points *= scale
  return points

class StupidSimpleLossMetric():
  def __init__(self):
    self.losses = [] # empty python array
    self.name = "StupidSimpleLossMetric" 
  def __call__(self, loss):
    self.losses.append(loss)
  def result(self):
    return sum(self.losses) / len(self.losses)
  def reset_state(self):
    self.losses = []

#train_loss = StupidSimpleLossMetric()
loss_tracker = tf.keras.metrics.Mean(name="loss")

class MobileHand(Model):
  def __init__(self, image_size, image_channels, mano_dir, T=3, **kwargs):
    super().__init__(**kwargs)
    self.image_size = image_size
    self.image_channels = image_channels
    self.mano_dir = mano_dir
    self.mobile_net = MobileNetV3Small(self.image_size, self.image_channels)
    self.T = T
    self.reg_module = IterativeRegression(59, 0.4, T)
    self.mano_model = MANO_Model(self.mano_dir)
    self.U = self.mano_model.U
    self.L = self.mano_model.L
  
  def call(self, x, iter=2, training=False):
    bs = tf.shape(x)[0]
    m_output = self.mobile_net(x)
    reg_output = self.reg_module(m_output, training)
    if training:
      reg_output = reg_output[iter]
    beta = tf.slice(reg_output, tf.constant([ 0, 0 ]), tf.stack([ bs, 10 ]))
    pose = tf.slice(reg_output, tf.constant([ 0, 10 ]), tf.stack([ bs, 48 ]))
    scale = tf.slice(reg_output, tf.constant([ 0, 58 ]), tf.stack([ bs, 1 ]))  
    scale = tf.expand_dims(scale, axis=1) # new shape is [bs, 1, 1]
    mano_mesh, mano_keypoints = self.mano_model(beta, pose, training)
    return (beta, pose, mano_mesh, mano_keypoints, scale)

  def train_step(self, data):

    input, gt = data 

    for t in range(self.T):
      with tf.GradientTape() as tape:
        beta, pose, mesh, keypoints, scale = self.call(input, iter=t, training=True)
        # This is the thing that takes our MANO template to the same shape as gt.
        gt_scale = tf.sqrt(tf.reduce_sum(tf.square(gt[:, 0] - gt[:, 8]), axis=1, keepdims=True)) / 0.0906426
        gt_scale = tf.expand_dims(gt_scale, axis=1) # should have shape = [bs, 1, 1]
        # apply regularization to keep corrections having estimates be on the manifold of valid hands.
        loss = LOSS(beta, pose, self.L, self.U, scale, keypoints, gt, gt_scale) if t == self.T - 1 else \
          LOSS2(beta, pose, self.L, self.U)
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      

    loss_tracker.update_state(loss)
    return {'loss': loss_tracker.result()}
  
  @property
  def metrics(self):
    return [loss_tracker]

# TODO: Rename this to L2.
# works for [bs, point_count, 3]
def mse(pred, gt):
  bs = tf.shape(pred)[0]
  point_count = tf.shape(pred)[1]
  mse = tf.reduce_sum(tf.reduce_sum(tf.square(pred - gt), axis=2), axis=1) / tf.cast(point_count, dtype=tf.float32)
  # loss = tf.reduce_sum(mse, axis=0) / bs 
  return mse

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

'''
NO WAY I THINK THE REGRESSION IS WRONG :)))))))

Like, there is this thing I do where we are like "We don't care about the regularization w.r.t 
to the root rotation.

'''


# beta is a tensor with shape of [bs, 10]. These are the estimated beta parameters that get fed to MANO.
# pose is a tensor with shape of [bs, 48]. These are the estimated pose parameters that get fed to MANO.
def LOSS_REG(beta, pose, L, U):
  # U and L are upper and lower limits for the theta MANO params.
  # U and L are only shape R^45.
  
  bs = tf.shape(pose)[0]
  point_count = tf.shape(pose)[1]
  # shape of pose is [bs, 48]
  pose = pose[ :, 3: ] # [bs, 45]
  #pose = tf.reshape(pose, (bs, -1, 3)) # new shape is [bs, 15, 3]

  L = tf.repeat(L, axis=0, repeats=bs) # [bs, 15, 3]
  U = tf.repeat(U, axis=0, repeats=bs) # [bs, 15, 3]

  # this is fine. 10 dimenionsal vector.
  loss = tf.reduce_sum(tf.square(beta), axis=1) + tf.reduce_sum(tf.square(pose), axis=1)

  '''loss += tf.reduce_sum(tf.reduce_sum(tf.square(
    tf.math.maximum(L - pose, tf.zeros(pose.shape)) + \
    tf.math.maximum(pose - U, tf.zeros(pose.shape))
  ), axis=2), axis=1) / point_count
  '''

  return loss

alpha_reg = 1e-3

# Master loss function
def LOSS(beta, pose, L, U, scale, pred, gt, gt_scale):
  gt_prime = gt / gt_scale # Inverse scale transform.
  return 1e2 * tf.reduce_mean(LOSS_2D(scale, pred, gt) + LOSS_3D(pred, gt_prime) + \
    LOSS_CAM(scale, gt_scale) + alpha_reg * LOSS_REG(beta, pose, L, U) ) # + 1e-1 * LOSS_REG(beta, pose, L, U)) # 

def LOSS2(beta, pose, L, U):
  return 1e2 * tf.reduce_mean( alpha_reg * LOSS_REG(beta, pose, L, U))


# input shape of arr1/arr2 is [bs, 21, 3]
def distance(arr1, arr2):
  diff = tf.math.subtract(arr1, arr2)
  distance = tf.norm(diff, axis = 2)
  distance = tf.squeeze(distance) # output shape is [bs, 21]
  return distance
