from __future__ import print_function

import numpy as np
import pylab as plt
import timeit
from bhtsne import bh_tsne

import zipfile
import brain.globals as st
from brain.data import extract_mat, normalize


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


def get_data(windows, datatype='eeg', trials_from=1, trials_to='end'):
    """ Get all data out of a given window and specified datatype as one concatenated numpy array

            :type windows: matlab struct
            :param windows: matlab struct win that contains all data of all trials of one window

            :type datatype: string
            :param datatype: Specifies what kind of data shall be extracted, e.g. 'eeg' or 'emg'
            """

    print('Assembling %s-data...' % datatype)
    data = None
    trials = None
    led = None
    data_length = 0

    for trial, win in enumerate(windows):
        if((trial+1 >= trials_from) and ((trials_to == 'end') or (trial+1 <= trials_to))):
            data_t = win.get(datatype + '_t')
            led_on = np.array([win.get('LEDon'), win.get('LEDoff')])

            data_temp = win.get(datatype)
            trials_temp = np.ones((data_temp.shape[0])) * (trial + 1)
            led_temp = np.where((data_t > led_on[0]) & (data_t < led_on[1]))[0] + data_length

            if(data is None):
                data = np.array(data_temp)
                trials = np.array(trials_temp)
                led = np.array(led_temp)
            else:
                data = np.vstack((data, data_temp))
                trials = np.hstack((trials, trials_temp))
                led = np.hstack((led, led_temp))

            data_length = data_length + data_temp.shape[0]

    return (data, trials, led)


def shuffle(data):
    """ Shuffle around data so that its rows are ordered in a random fashion

            :type data: numpy array
            :param data: input data to be shuffled
            """

    shuffle = np.arange(data.shape[0])
    np.random.shuffle(shuffle)
    undo_shuffle = np.argsort(shuffle)

    return (data[shuffle], undo_shuffle)



if __name__ == '__main__':

    # --- Adjust parameters of bh-tsne and set the dpi-value of the output image file ---

    #datatype: Specify if you want eeg or emg data
    datatype = 'eeg'
    #participant and series: Specify participant and series
    participant = 6
    series = 3
    #trials_from and trials_to: Specify which trials t-SNE shall run on
    trials_from = 1
    trials_to = 'end'
    #p: perplexity
    p = 30
    #t: theat value
    t = 0.5
    #dpi: quality of generated plots
    dpi = 500
    #directory: directory in which the plot ought to be saved
    directory = 'plots/tsne'
    #randomize: Shuffle the data to overcome bh-tsne weak points
    randomize = True
    #normalize: Specify of data ought to be normalized before being applied on t-SNE
    normalize_data = True

    # -----------------------------------------------------------------------------------

    #Fetching data
    ws = get_ws(participant=participant, series=series)
    windows = ws.get('win')
    (data, trials, led) = get_data(windows, datatype=datatype, trials_from=trials_from, trials_to=trials_to)
    n = data.shape[0]

    #Run bh-tsne
    data = data[:n]
    if normalize_data:
        print('Normalizing data...')
        data[...] = normalize(data)
    if randomize:
        print('Shuffling data...')
        (data[...], undo_shuffle) = shuffle(data)
    start_time = timeit.default_timer()
    Y = run_bhtsne(data, theta=t, perplexity=p)
    if randomize:
        Y[...] = Y[undo_shuffle]
    end_time = timeit.default_timer()

    print('bh-t-SNE ran for %f minutes' % ((end_time - start_time) / 60))

    #Create scatter plots
    print('Creating scatter plots...')
    plt.title('%s t-SNE' % datatype)

    #Separating data into points where the led was on and off respectively
    led = led[np.where(led < n)]
    trials = trials[:n]

    mask = np.ones(Y.shape[0], dtype=bool)
    mask[led] = False

    Y_led_on = Y[led]
    Y_led_off = Y[mask]
    trials_led_on = trials[led]
    trials_led_off = trials[mask]

    #Plot data: Data points where the led was on are marked with a black edge
    #Data points where the led was off have no edge color
    plt.scatter(Y_led_off[:, 0], Y_led_off[:, 1], s=10, c=trials_led_off, marker='o', edgecolors='none')
    plt.scatter(Y_led_on[:, 0], Y_led_on[:, 1], s=6, c=trials_led_on, marker='o', edgecolors='black')

    file = directory + '/' + datatype + str(n) + '_P' + str(participant) + '_S' + str(series) + '_p' + str(p)\
           + '_t' + str(t) + '_r' + str(randomize) + '_n' + str(normalize_data) + '_dpi' + str(dpi) + '.png'
    plt.savefig(file, bbox_inches='tight', dpi=dpi)