#This file is for parsing the data so that we only have single hand images
#make sure that the folder 'Single_hand_data' is in this directory and make sure that the folders 'evaluation' and 'training
#are in that folder. Each of these folders should have a 'mask' and 'color' folder
import os
import pickle
import imageio
import numpy as np
import time

dir = 'RHD_small' #change this to the name of the dataset folder
sets = ['training','evaluation']
for set in sets:
    path = os.path.join(dir, set)
    with open(os.path.join(path, 'anno_%s.pickle' % set), 'rb') as fi:
        anno_all = pickle.load(fi)

    
    total_training_examples = 41257 + 1
    if dir == 'RHD_small':
        total_training_examples = 203
    valid_training_examples = 0
    start_time = time.time()
    path2 = 'Single_hand_data'
    print("Begin single hand parse")
    for  sample_id,anno in anno_all.items():
        if dir == "RHD_small":
            if sample_id>203:
                break
        # format of the kp_visible array
        '''
        # 0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
        # 21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
        '''
        kp_visible = (anno['uv_vis'][:, 2] == 1)
        case1 = np.sum(kp_visible[0:21])
        case2 = np.sum(kp_visible[21:])
        #print("kp_visible", kp_visible)
        #print("case1", case1)
        #print("case2", case2)

        # also invalidates training examples where none of the hands can be seen. We must see at least some of just one hand for it to be valid.
        valid_case = (case1 > 0 and case2 == 0) or (case1 == 0 and case2 > 0) 
        if (valid_case):
            valid_training_examples += 1
            image = imageio.imread(os.path.join(path, 'color', '%.5d.png' % sample_id))
            mask = imageio.imread(os.path.join(path, 'mask', '%.5d.png' % sample_id))            
            filename = "%.5d.png" % sample_id
            print(filename)
            imageio.imwrite(os.path.join(path2,set,'color',filename),image)
            imageio.imwrite(os.path.join(path2,set,'mask',filename),mask)         
            




    end_time = time.time()
    print("Total elapsed time for single hand parse =", end_time - start_time, "s")
    print("valid %s examples:" % set, valid_training_examples)
    print("Amount of %s training examples = " % set, valid_training_examples / total_training_examples * 100, "%")

    
    


