'''
CS688 HW04: Restricted Boltzmann Machine for handwritten digit recognition

Question 4: Feature extraction and classification

@author: Emma Strubell
'''

import load_data as load
import rbm

train_size = load.max_train_size
test_size = load.max_test_size

print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)

print "Loading %d/%d training labels" % (train_size, load.max_train_size)
train_labels = load.load_labels('train', train_size)

print "Loading %d/%d training instances" % (test_size, load.max_test_size)
test_instances = load.load_data('test', test_size)

print "Loading %d/%d training labels" % (test_size, load.max_test_size)
test_labels = load.load_labels('test', test_size)

print "Loading model parameters"
w_c = load.load_params("C400")
w_b = load.load_params("B400")
w_p = load.load_params("P400")

def q4a():
    # compute embeddings for train and test data
    print "Computing embeddings"
    train_embeddings = rbm.compute_embeddings(w_c, w_b, w_p, train_instances)
    test_embeddings = rbm.compute_embeddings(w_c, w_b, w_p, test_instances)
    
    # write train and test embeddings as labeled feature vectors for SVMLight
    print "Writing labeled feature vectors (embeddings) for SVMLight"
    load.write_svmlight(train_embeddings, train_labels, "train-embeddings")
    load.write_svmlight(test_embeddings, test_labels, "test-embeddings")

def q4b():
    # write train and test embeddings as labeled feature vectors for SVMLight
    print "Writing labeled feature vectors (raw) for SVMLight"
    load.write_svmlight(train_instances, train_labels, "train-raw")
    load.write_svmlight(test_instances, test_labels, "test-raw")

# generate feature vectors for SVMLight using RBM embeddings
q4a()

# generate feature vectors for SVMLight using raw binary image data
q4b()