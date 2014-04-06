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

t = 50  # number of training iterations
k = 400 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
alpha = 0.1  # step size
lam = 0.0001 # regularization param

def q3ab():
    w_c, w_b, w_p, results, hiddens = rbm.train_rbm(train_instances, t, k, b, c, alpha, lam)
    
    # plot final samples for each Gibbs chain
    plotter.plot100(results)
    
    # plot all receptive fields
    num_plots = k//100
    for i in range(num_plots):
        plotter.plot100(w_p[i*100:(i+1)*100])
    
    # write model to files
    load.write_params(w_c, "MNISTWC400")
    load.write_params(w_b, "MNISTWB400")
    load.write_params(w_p, "MNISTWP400")

q3ab()
plt.show()