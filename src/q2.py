'''
CS688 HW04: Restricted Boltzmann Machine for handwritten digit recognition

Question 2: Block Gibbs sampling

@author: Emma Strubell
'''

import numpy as np
import load_data as load
import matplotlib.pyplot as plt
import plotter
import rbm

print "Loading P params"
w_p = load.load_params('P')

print "Loading B params"
w_b = load.load_params('B')

print "Loading C params"
w_c = load.load_params('C')

iterations = 500
d, k = w_p.shape

def q2a():
    interval = 5
    results = rbm.block_gibbs_sample(w_p, w_b, w_c, d, k, iterations, verbose=True)
    plotter.plot100(results[0][::interval])

def q2bc():
    num_chains = 100
    results = np.array([rbm.block_gibbs_sample(w_p, w_b, w_c, d, k, iterations, i) for i in range(num_chains)])
    images = results[:,0]
    energies = results[:,1]
    plotter.plot100(images[:,-1])
    plotter.plot_energies(energies[:5])

q2a()

q2bc()

plt.show()