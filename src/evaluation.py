# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory that contains every single evaluation image.
# download_image is a function that we can call (we pass it one param: fileName), and 
#   it will return to us a numpy array for the image.
def time_model(model, rhd_eval_dir, download_image):
    pass


# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory name for a directory 
#   that contains every single evaluation image (of just single hands).
# download_image is a function that we can call.
# y_test is a numpy array with the 3D keypoints for every single image in the RHD evaluation set.
#   the indices into this array are the names of the image files.
def evaluate_model(model, rhd_eval_dir, download_image, y_test):
    pass

# checkpoint_dir is a directory that contains the model checkpoints to iterate over.
# model is a param for the callable tensorflow model (with loaded weights).
# rhd_eval_dir is a directory that contains every single evaluation image (of just single hands).
# download_image is a function that we can call (we pass it one param: fileName), and 
#   it will return to us a numpy array for the image.
# y_test is a numpy array with the 3D keypoints for every single image in the RHD evaluation set.
#   the indices into this array are the names of the image files.
def generate_loss_graph(checkpoint_dir, model, rhd_eval_dir, download_image, y_test):
    pass


