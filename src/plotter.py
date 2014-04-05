import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot100(images):
    # plot a 10x10 array of 28x28 images
    xplots = 10
    yplots = 10
    fig, ax = plt.subplots(xplots, yplots)
    for i in range(xplots):
        for j in range(yplots):
            ax[i][j].axis('off')
            ax[i][j].imshow(images[i*10+j].reshape((28,28)), interpolation='nearest', cmap='gray')

def plot1(im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im.reshape((28,28)), interpolation='nearest', cmap='gray')

def plot_energies(energies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Energy vs. Sampling Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    for es in energies:
        ax.plot(es)