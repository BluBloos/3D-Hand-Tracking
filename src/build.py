from _thread import *
from marshal import load
import time
from importlib import reload

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
img_count = 2350
qmindlib.init(rhd_dir, BATCH_SIZE, img_count)
print(cstr("Loading train_ds with {} many images".format(img_count)))
train_ds = qmindlib.train_ds
eval_ds = qmindlib.eval_ds
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
model.compile(optimizer="adam")

def get_ckpt_state():
    return int(open("model_state.txt", "r").read())

def set_ckpt_state(ckpt):
    f = open("model_state.txt", "w")
    f.write(str(ckpt))

def load_model_cpkt(ckpt):
    if ckpt > 0:
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
    print("rm                                     - Reload model.");
    print("lc<ckpt>                               - Loads ckpt into model.");
    print("sdic <count>                           - Prepares tf.data.Dataset with count many images.")
    print("help                 (h)               - Print all commands.");
    print("exit                 (e)               - Exit the interactive system.");
   
print_help()

def train_loop(epochs, lr):

    EPOCHS = epochs
    model.optimizer.learning_rate = lr # should work? Hope so.
    LAST_CHECKPOINT = get_ckpt_state()

    # TODO(Noah): need to add the checkpoint saving.
    model.fit(train_ds, epochs=EPOCHS)


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
        evaluation.evaluate_model(model, eval_ds)
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
    elif user_command == "rm":
        launch_loader() 
        tf.keras.backend.clear_session() # presumably this does nothing to our train_ds
        reload(mobilehand)
        model = mobilehand.MobileHand(IMAGE_SIZE, IMAGE_CHANNELS, MANO_DIR, T=T)
        model.mobile_net.freeze()
        model.compile(optimizer="adam")
        set_ckpt_state(0)
        stop_loader()
    elif user_command == "vis":
        pass
    elif user_command == "help" or user_command == "h":
        print_help()
    elif user_command == "exit" or user_command == "e":
        exit() 
