from _thread import *
from marshal import load
import time

SHOULD_LOAD = False
loader_lock = allocate_lock()

def loader():
    global SHOULD_LOAD
    sequence = ['-', '\\', '|', '/', '-', '\\', '|', '/']
    iter = 0
    while SHOULD_LOAD:
        loader_lock.acquire()
        print("\r{} ".format(sequence[iter]), end="")
        iter += 1
        iter = iter % len(sequence)
        loader_lock.release() 
        time.sleep(1/5) # 1 s = 1000ms
    print()

def launch_loader():
    global SHOULD_LOAD
    SHOULD_LOAD = True
    print()
    start_new_thread(loader, ())

def stop_loader():
    global SHOULD_LOAD
    loader_lock.acquire()
    SHOULD_LOAD = False
    loader_lock.release()
    time.sleep(1/3)
    
print("\nWelcome to our " + '\033[92m' + "nifty" + '\033[0m' + " machine learning framework!")
print("Initializing qmind_lib")

launch_loader()

BATCH_SIZE = 32
import os
import numpy as np
import tensorflow as tf
#tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__)
import random
import evaluation
from qmind_lib import cstr, rstr, set_color_cyan, set_color_normal
import qmind_lib as qmindlib
rhd_dir = os.path.join("..", "SH_RHD")
qmindlib.init(rhd_dir, BATCH_SIZE, img_count=2350)
y_train = qmindlib.y_train
y_test = qmindlib.y_test
k_train = qmindlib.k_train
k_test = qmindlib.k_test
y2_train = qmindlib.y2_train
y2_test = qmindlib.y2_test
train_ds = qmindlib.train_ds
MANO_DIR = os.path.join("..", "mano_v1_2")
import mobilehand
T = 3
# TODO(Noah): Pretty sure that there is something to do with IMAGE_SIZE and how it's not actually
# being used when creating the dataset (which it should be).
IMAGE_SIZE = 224
GRAYSCALE = False
IMAGE_CHANNELS = 1 if GRAYSCALE else 3
model = mobilehand.MobileHand(IMAGE_SIZE, IMAGE_CHANNELS, MANO_DIR, T=T)
model.mobile_net.freeze()
U = model.mano_model.U
L = model.mano_model.L
loss_fn = mobilehand.LOSS
loss_fn2 = mobilehand.LOSS2

# TODO(Noah): We probably want to use the model.fit function! This just seems to be the best idea overall.
# It's gonna give us a GOOD printing routine, and we do not need to predefine this ourselves.
class StupidSimpleLossMetric():
  def __init__(self):
    self.losses = [] # empty python array 
  def __call__(self, loss):
    self.losses.append(loss)
  def result(self):
    return sum(self.losses) / len(self.losses)
  def reset_states(self):
    self.losses = []

optimizer = tf.keras.optimizers.Adam() # defaults should work just fine
train_loss = StupidSimpleLossMetric()

@tf.function
def train_step(input, gt):
  for t in range(T):
    with tf.GradientTape() as tape:
      beta, pose, mesh, keypoints, scale = model(input, iter=t, training=True)
      # This is the thing that takes our MANO template to the same shape as gt.
      gt_scale = tf.sqrt(tf.reduce_sum(tf.square(gt[:, 0] - gt[:, 8]), axis=1, keepdims=True)) / 0.0906426
      gt_scale = tf.expand_dims(gt_scale, axis=1) # should have shape = [bs, 1, 1]
      # apply regularization to keep corrections having estimates be on the manifold of valid hands.
      loss = loss_fn(beta, pose, L, U, scale, keypoints, gt, gt_scale) if t == T - 1 else loss_fn2(beta, pose, L, U)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def get_ckpt_state():
    return int(open("model_state.txt", "r").read())

def set_ckpt_state(ckpt):
    f = open("model_state.txt", "w")
    f.write(str(ckpt))

def load_model_cpkt(ckpt):
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{:04d}.ckpt".format(ckpt))
    model.load_weights(checkpoint_path)
    print(cstr("Loaded model checkpoint {}".format(ckpt)))

# Load in model checkpoint from the current state.
ckpt = get_ckpt_state()
checkpoint_dir = os.path.join('../','checkpoints')
load_model_cpkt(ckpt)

stop_loader()

def print_help():
    print("");
    print(cstr("=== Commands ==="));
    print("train <epochs> <lr>  (t)               - Train the model.");
    print("eval                 (ev)              - Generate the 3D PCK curve + timing metrics.");
    print("vis                  (v)               - Run the visualizer program.");
    print("lc<ckpt>                               - Loads ckpt into model.");
    print("sdic <count>                           - Prepares tf.data.Dataset with count many images.")
    print("help                 (h)               - Print all commands.");
    print("exit                 (e)               - Exit the interactive system.");
   
print_help()

def train_loop(epochs, lr):

    EPOCHS = epochs
    optimizer.learning_rate = lr
    LAST_CHECKPOINT = get_ckpt_state()

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        print("Begin epoch", epoch)
        train_loss.reset_states()
        start = time.time()
        
        for img, y in train_ds:
            loss = train_step(img, y)
            train_loss(loss.numpy())    

        end = time.time()

        print(
            f'Epoch {epoch}, '
            f'Time {end-start} s'
            f'Loss: {train_loss.result()}, '
        )

        # Save the model parameters
        ckpt_index = LAST_CHECKPOINT + epoch
        checkpoint_filepath = os.path.join(checkpoint_dir, "cp-{:04d}.ckpt".format(ckpt_index))
        model.save_weights(checkpoint_filepath)
        set_ckpt_state(ckpt_index)
        print(cstr("Saved weights to {}".format(checkpoint_filepath)))


while (1):

    print("\n> ", end="");
    set_color_cyan()
    _user_command = input()
    args = _user_command.split(" ")
    user_command = args[0] 
    set_color_normal()

    # process the user command
    if user_command == "train" or user_command == "t":
        if len(args) != 3:
            print(rstr("Invalid way to do train command"))
        else:
            epochs = int(args[1])
            lr = float(args[2])
            train_loop(epochs, lr)
    elif user_command == "eval" or user_command == "ev":
        launch_loader()    
        evaluation.evaluate_model(model, train_ds)
        stop_loader()
    elif user_command.startswith("lc"):
        ckpt_index = int(user_command[2:])
        load_model_cpkt(ckpt_index)
        set_ckpt_state(ckpt_index)
    elif user_command == "sdic":
        if len(args) != 2:
            print(rstr("Invalid way to do sdic command"))
        else:
            launch_loader()  
            img_count = int(args[1])
            img_dir = os.path.join(rhd_dir, "training", "color")
            qmindlib.create_tf_dataset(img_dir, BATCH_SIZE, img_count)
            train_ds = qmindlib.train_ds
            stop_loader()
    elif user_command == "vis":
        pass
    elif user_command == "help" or user_command == "h":
        print_help()
    elif user_command == "exit" or user_command == "e":
        exit() 
