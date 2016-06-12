from __future__ import print_function

import numpy as np
import pylab as plt
import timeit
from bhtsne import bh_tsne

import zipfile
import brain.data.globals as st
from brain.data.util import extract_mat, _todict


'''
Runs the bh-tsne
'''
def run_bhtsne(data_set, theta=0.5, perplexity=50):
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


'''
Returns the 'ws'-struct of a WS_P*_S*.mat - file
for a given participant and series
'''
def get_ws(participant=1, series=1):
    archive = '../' + st.DATA_PATH + 'P' + str(participant) + '.zip'
    print('Reading ' + archive + '...')
    f_zip = zipfile.ZipFile(archive, 'r')
    f_mat = 'P' + str(participant) + '/WS_P'+str(participant) + '_S' + str(series) + '.mat'
    mat = extract_mat(f_zip, f_mat, relative_path='../'+st.DATA_PATH)
    return mat.get('ws')


def get_data(windows):
    print('Assembling EEG- and EMG-data...')
    eeg_shape = np.asarray([0, 32])
    emg_shape = np.asarray([0, 5])
    for win in windows:
        win = _todict(win)
        eeg_shape[0] += win.get('eeg').shape[0]
        emg_shape[0] += win.get('emg').shape[0]

    EEG = np.empty(eeg_shape)
    EEG_trial = np.empty(eeg_shape[0])
    EMG = np.empty(emg_shape)
    EMG_trial = np.empty(emg_shape[0])
    eeg_i = 0
    emg_i = 0
    for trial, win in enumerate(windows):
        win = _todict(win)
        eeg = win.get('eeg')
        emg = win.get('emg')
        EEG[eeg_i:eeg_i + eeg.shape[0], :] = eeg
        EEG_trial[eeg_i:eeg_i + eeg.shape[0]] = np.ones((eeg.shape[0])) * (trial + 1)
        EMG[emg_i:emg_i + emg.shape[0], :] = emg
        EMG_trial[emg_i:emg_i + emg.shape[0]] = np.ones((emg.shape[0])) * (trial + 1)
        eeg_i += eeg.shape[0]
        emg_i += emg.shape[0]

    return (EEG, EEG_trial, EMG, EMG_trial)


if __name__ == '__main__':
    #Read data for a specific participant and series and concatenate eeg and emg
    ws = get_ws(participant=1, series=1)
    windows = ws.get('win')
    (eeg, eeg_trial, emg, emg_trial) = get_data(windows)

    #Adjust parameters of bh-tsne and set the dpi-value of the output image file
    n = eeg.shape[0]
    p = 30
    t = 0.5
    dpi = 500

    #Run bh-tsne
    start_time = timeit.default_timer()
    Y_eeg = run_bhtsne(eeg[:n], theta=t, perplexity=p)
    #Y_emg = run_bhtsne(emg[:n], theta=t, perplexity=p)
    end_time = timeit.default_timer()

    print('bh-t-SNE ran for %f minutes' % ((end_time - start_time) / 60))

    #Create scatter plots
    print('Creating scatter plots...')
    plt.title('EEG t-SNE')
    plt.scatter(Y_eeg[:, 0], Y_eeg[:, 1], 10, eeg_trial[:n], edgecolors=None, marker='.')
    #plt.legend(loc='upper left', numpoints=1)
    file = 'eeg' + str(n) + '_p' + str(p) + '_t' + str(t) + '_dpi' + str(dpi) + '.png'
    plt.savefig(file, bbox_inches='tight', dpi=dpi)

    '''plt.title('EMG t-SNE')
    plt.scatter(Y_emg[:, 0], Y_emg[:, 1], 10, emg_trial[:n], edgecolors=None, marker='.')
    # plt.legend(loc='upper left', numpoints=1)
    file = 'emg' + str(n) + '_p' + str(p) + '_t' + str(t) + '_dpi' + str(dpi) + '.png'
    plt.savefig(file, bbox_inches='tight', dpi=dpi)'''