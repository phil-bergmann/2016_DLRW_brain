from __future__ import print_function

import numpy as np
import pylab as plt
import timeit
from bhtsne import bh_tsne

import zipfile
import brain.data.globals as st
from brain.data.util import extract_mat, _todict


def run_bhtsne(data_set, theta=0.5, perplexity=50):
    """ Runs the bh-tsne on the given data

            :type data_set: numpy array
            :param data_set: Numpy array on which bh-tsne shall be run

            :type theta: float
            :param theta: Specifies the theta parameter

            :type perplexity: int
            :param perplexity: Specifies the perplexity
            """

    n = data_set.shape[0]
    print('Running Barnes-Hut - t-SNE on %d data points...' % n)
    data_bhtsne = np.zeros((n, 2))

    for dat, temp in zip(bh_tsne(np.copy(data_set), theta=theta, perplexity=perplexity), data_bhtsne):
        temp[...] = dat

    print('\nNormalizing...')
    min = np.min(data_bhtsne, axis=0)
    data_bhtsne = data_bhtsne - min
    max = np.max(data_bhtsne, axis=0)
    data_bhtsne = data_bhtsne / max

    return data_bhtsne


def get_ws(participant=1, series=1):
    """ Returns the 'ws'-struct of a WS_P*_S*.mat - file for a given participant and series

            :type participant: int
            :param participant: Specifies for which participant data shall be returned

            :type series: int
            :param series: Specifies for which series data shall be returned
            """

    archive = '../' + st.DATA_PATH + 'P' + str(participant) + '.zip'
    print('Reading ' + archive + '...')
    f_zip = zipfile.ZipFile(archive, 'r')
    f_mat = 'P' + str(participant) + '/WS_P'+str(participant) + '_S' + str(series) + '.mat'
    mat = extract_mat(f_zip, f_mat, relative_path='../'+st.DATA_PATH)
    return mat.get('ws')


def get_data(windows, datatype='eeg'):
    """ Get all data out of a given window and specified datatype as one concatenated numpy array

            :type windows: matlab struct
            :param windows: matlab struct win that contains all data of all trials of one window

            :type datatype: string
            :param datatype: Specifies what kind of data shall be extracted, e.g. 'eeg' or 'emg'
            """

    print('Assembling %s-data...' % datatype)
    data = None
    trials = None

    for trial, win in enumerate(windows):
        win = _todict(win)
        data_temp = win.get(datatype)
        trials_temp = np.ones((data_temp.shape[0])) * (trial + 1)
        if(data is None):
            data = np.array(data_temp)
            trials = np.array(trials_temp)
        else:
            data = np.vstack((data, data_temp))
            trials = np.hstack((trials, trials_temp))

    return (data, trials)




if __name__ == '__main__':
    #Read data for a specific participant and series and concatenate it into one numpy array
    datatype = 'eeg'
    ws = get_ws(participant=1, series=1)
    windows = ws.get('win')
    (data, trials) = get_data(windows, datatype=datatype)

    #Adjust parameters of bh-tsne and set the dpi-value of the output image file
    n = 5000#eeg.shape[0]
    p = 30
    t = 0.5
    dpi = 500

    #Run bh-tsne
    start_time = timeit.default_timer()
    Y = run_bhtsne(data[:n], theta=t, perplexity=p)
    end_time = timeit.default_timer()

    print('bh-t-SNE ran for %f minutes' % ((end_time - start_time) / 60))

    #Create scatter plots
    print('Creating scatter plots...')
    plt.title('%s t-SNE' % datatype)
    plt.scatter(Y[:, 0], Y[:, 1], s=10, c=trials[:n], marker='o', edgecolors='none')
    #plt.legend(loc='upper left', numpoints=1)
    file = datatype + str(n) + '_p' + str(p) + '_t' + str(t) + '_dpi' + str(dpi) + '.png'
    plt.savefig(file, bbox_inches='tight', dpi=dpi)