'''
CS688 HW04: Restricted Boltzmann Machine for handwritten digit recognition

Question 3: RBM training

@author: Emma Strubell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import load_data as load
import plotter
import rbm

train_size = load.max_train_size
test_size = load.max_test_size
 
print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)
 
#print "Loading %d/%d training labels" % (train_size, load.max_train_size)
#train_labels = load.load_labels('train', train_size)

t = 50  # number of training iterations
k = 400 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
alpha = 0.1  # step size
lam = 0.0001 # reguarlization param

def q3ab():
    _, _, w_p, results = rbm.train_rbm(train_instances, t, k, b, c, alpha, lam)
    
    # plot final samples for each Gibbs chain
    plotter.plot100(results)
    
    # plot all receptive fields
    print w_p.shape
    num_plots = k//100
    for i in range(num_plots):
        plotter.plot100(w_p[i*100:(i+1)*100])

q3ab()

plt.show()