from __future__ import print_function

import numpy as np
import pylab as plt
import timeit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import matplotlib.colors
from copy import copy
from bhtsne import bh_tsne
from sys import stdout


def mnist_scatter(x, y, data_set, ax=None, zoom=1):
    print('Creating Scatterplot...')
    if ax is None:
        ax = plt.gca()

    artists = []
    i = 0
    for x0, y0 in zip(x, y):
        digit = np.reshape(data_set[i,:], (28, 28))
        digit = np.ones((28, 28)) - digit

        mnist_norm = matplotlib.colors.Normalize(vmin=.10, vmax=0.90, clip=False)
        mnist_cmap = copy(cm.get_cmap('gray'))
        mnist_cmap.set_over('w', alpha=0)

        im = OffsetImage(digit, zoom=zoom, norm=mnist_norm, cmap=mnist_cmap, interpolation='none')
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        i += 1
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists



def run_bhtsne(data_set, theta=0.5, perplexity=50):
    n = data_set.shape[0]
    print('Running Barnes-Hut - t-SNE on %d MNIST-digits...' % n)
    data_bhtsne = np.zeros((n, 2))

    for dat, temp in zip(bh_tsne(np.copy(data_set), theta=theta, perplexity=perplexity), data_bhtsne):
        temp[...] = dat

    print('\nNormalizing...')
    min = np.min(data_bhtsne, axis=0)
    data_bhtsne = data_bhtsne - min
    max = np.max(data_bhtsne, axis=0)
    data_bhtsne = data_bhtsne / max

    return data_bhtsne



if __name__ == '__main__':
    n = 70000
    p = 30
    t = 0.5

    train_set, valid_set, test_set = loadMNIST('mnist.pkl.gz')
    x = np.concatenate((train_set[0], valid_set[0], test_set[0]))

    start_time = timeit.default_timer()
    Y = run_bhtsne(x[:n], theta=t, perplexity=p)
    mnist_scatter(Y[:, 0], Y[:, 1], x[:n])
    end_time = timeit.default_timer()

    print('t-SNE ran for %f minutes' % ((end_time - start_time) / 60))

    #plt.show()

    print('Writing results to file...')
    for i in range(7, 10):
        dpi = i * 100
        print('Writing with %d dpi' % dpi)
        file = 'bhtsne_n'+str(n)+'_p'+str(p)+'_t'+str(t)+'_dpi'+str(dpi)+'.png'
        plt.savefig(file, bbox_inches='tight', dpi=dpi)