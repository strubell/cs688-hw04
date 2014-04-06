'''
CS688 HW03: Monte Carlo De-noising

Question 1: Monte Carlo De-noising for binary images

@author: Emma Strubell
'''

import numpy as np

max_train_size = 60000
max_test_size = 10000

data_dir = "../data/"
model_dir = "../models/"

data_prefix = data_dir + "MNIST"
model_prefix = model_dir + "MNISTW"

# load trained parameter (model) files
def load_params(type): return np.loadtxt("%s%s.txt" % (model_prefix, type))

# load data into numpy array
def load_data(type, num_lines=0): 
    return np.genfromtxt("%sX%s.txt" % (data_prefix, type), skip_footer=(max_train_size if type == "train" else max_test_size)-num_lines)

# load data labels into numpy array
def load_labels(type, num_lines=0):
    return np.genfromtxt("%sY%s.txt" % (data_prefix, type), skip_footer=(max_train_size if type == "train" else max_test_size)-num_lines)

# write trained parameter (model) files
def write_params(vec, name): np.savetxt(model_dir+name+".txt", vec)

# write labeled feature vectors in SVMlight sparse format
def write_svmlight(features, labels, fname):
    file = open(data_dir+fname, 'w')
    for feats,label in zip(features,labels):
        file.write("%d " % (label))
        for idx,feat in enumerate(feats):
            if(not feat == 0.0):
                file.write("%d:%g " % (idx+1,feat))
        file.write("\n")
    file.close()