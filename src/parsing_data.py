# Author: Max Vincent

import os
import pickle
import imageio
import numpy as np
import time

# This function assumes that out_dir exists with subfolders 'evaluation' and 'training', each of
# which has subfolder 'color'.
# NOTE(Noah): Set is either "evaluation" or "training"
def parse_dataset(set, total_training_examples, dataset_dir, out_dir):
  path = os.path.join(dataset_dir, set)
  with open(os.path.join(path, 'anno_%s.pickle' % set), 'rb') as fi:
      anno_all = pickle.load(fi)
  valid_training_examples = 0
  start_time = time.time()
  path2 = out_dir
  print("Begin single hand parse")
  for sample_id,anno in anno_all.items():
    if sample_id > total_training_examples:
      break
    # format of the kp_visible array
    '''
    # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
    '''
    kp_visible = (anno['uv_vis'][:, 2] == 1)
    case1 = np.sum(kp_visible[0:21])
    case2 = np.sum(kp_visible[21:])

    # also invalidates training examples where none of the hands can be seen. 
    # We must see at least some of just one hand for it to be valid.
    valid_case = (case1 > 0 and case2 == 0) or (case1 == 0 and case2 > 0) 
    if (valid_case):
      valid_training_examples += 1
      image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
      #mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))            
      filename = "%.5d.png" % sample_id
      print(filename)
      imageio.imwrite(os.path.join(path2,set,'color',filename),image)
      #imageio.imwrite(os.path.join(path2,set,'mask',filename),mask)         
            
  end_time = time.time()
  print("Total elapsed time for single hand parse =", end_time - start_time, "s")
  print("valid %s examples:" % set, valid_training_examples)
  print("Amount of %s training examples = " % set, valid_training_examples / total_training_examples * 100, "%")

    
    


