import matplotlib.pyplot as plt
import os

def plot_samples_2d(data, path=None, name=None):
    plt.figure(figsize=(5,5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.scatter(data[:, 0], data[:, 1]) #, s=15)
    
    if name:
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(path + name + ".png", format="png")
    else:
        plt.show()