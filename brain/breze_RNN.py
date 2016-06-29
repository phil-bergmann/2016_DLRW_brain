import cPickle
import math
import time

import numpy as np
import theano.tensor as T

from data import get_eeg_emg
import globals as st

from breze.learn.data import interleave, padzeros, split
from breze.learn import base
from breze.learn.rnn import SupervisedFastDropoutRnn, SupervisedRnn
import breze.learn.display as D

import climin.initialize
import climin.stops
import climin.mathadapt as ma

import matplotlib.pyplot as plt

def test_RNN(n_layers = 1, batch_size = 50):

    optimizer = 'rmsprop', {'step_rate': 0.0001, 'momentum': 0.9, 'decay': 0.9}
    #optimizer = 'adadelta', {'decay': 0.9, 'offset': 1e-6, 'momentum': .9, 'step_rate': .1}
    #optimizer = 'gd', {'step_rate': 1e-4, 'momentum': .99, 'momentum_type': 'nesterov'}
    #optimizer = 'adamdelta'


    n_hiddens = [100] * n_layers

    #from breze.arch.component.varprop.transfer import tanh

    m = SupervisedRnn(
        5, n_hiddens, 2,  out_transfer='sigmoid', loss='bern_ces',
        hidden_transfers=['tanh'] * n_layers,
        batch_size=batch_size,
        optimizer=optimizer)

    m.exprs['true_loss'] = m.exprs['loss']

    f_loss = m.function(['inpt', 'target'], 'true_loss')

    '''
    def test_nll():
        nll = 0
        n_time_steps = 0
        for x, z in zip(tx, tz):
            nll += f_loss(x[:, np.newaxis], z[:, np.newaxis]) * x.shape[0]
            n_time_steps += x.shape[0]
        return nll / n_time_steps

    '''

    #######################
    # LOAD DATA TO ARRAYS #
    #######################

    # IMPORTANT: Breze wants the data to be aligned (timesteps, samples, dimensionality)

    data = get_eeg_emg(1, 1)
    p_train = 0.66
    n_train = int(len(data) * p_train)
    n_val = len(data) - int(len(data) * p_train)
    X = np.zeros((st.N_EMG_TIMESTEPS, n_train, st.N_EMG_SENSORS))
    Z = np.zeros((st.N_EMG_TIMESTEPS, n_train, st.N_EMG_TARGETS))
    VX = np.zeros((st.N_EMG_TIMESTEPS, n_val, st.N_EMG_SENSORS))
    VZ = np.zeros((st.N_EMG_TIMESTEPS, n_val, st.N_EMG_TARGETS))
    for i in range(len(data)):
        j = 0
        for d in data[i]['emg_target'].iteritems():
            if i < n_train:
                X[j, i, ...] = d[1][0:st.N_EMG_SENSORS]
                Z[j, i, ...] = d[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS+st.N_EMG_TARGETS]
            else:
                VX[j, i-n_train, ...] = d[1][0:st.N_EMG_SENSORS]
                VZ[j, i-n_train, ...] = d[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS + st.N_EMG_TARGETS]
            j += 1

    climin.initialize.randomize_normal(m.parameters.data, 0, 0.01)

    max_passes = 10000
    max_minutes = 600
    max_iter = max_passes * X.shape[1] / m.batch_size
    batches_per_pass = int(math.ceil(float(X.shape[1]) / m.batch_size))
    pause = climin.stops.ModuloNIterations(batches_per_pass * 1)

    stop = climin.stops.Any([
        climin.stops.TimeElapsed(max_minutes * 60),
        # climin.stops.patience('val_loss', 1000, grow_factor=1.1, threshold=0.0001),
        climin.stops.NotBetterThanAfter(30, 100),
    ])

    start = time.time()
    # Set up a nice printout.
    header = '#', 'seconds', 'loss', 'val loss', 'test loss'
    print '\t'.join(header)

    def plot():
        figure, (axes) = plt.subplots(4, 1)

        x_axis = np.arange(st.N_EMG_TIMESTEPS)

        result = m.predict(VX[:, 0:1, :])

        axes[0].set_title("hand_move_target")
        axes[0].plot(x_axis, VZ[:, 0, 0])
        axes[1].set_title("grasp_target")
        axes[1].plot(x_axis, VZ[:, 0, 1])
        axes[2].set_title("hand_move_rnn")
        axes[2].plot(x_axis, result[:, 0, 0])
        axes[3].set_title("grasp_rnn")
        axes[3].plot(x_axis, result[:, 0, 1])

        figure.subplots_adjust(hspace=0.5)

        figure.savefig('test.png')

        plt.close(figure)


    infos = []

    for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause, True)):
        info['loss'] = float(info['loss'])
        info['val_loss'] = float(info['val_loss'])
        info['test_loss'] = 100#float(ma.scalar(test_nll()))

        #    if info['test_loss'] < 8.58:
        #        break

        info.update({
            'time': time.time() - start,
            # 'spectral_radius': get_spectral_radius(m.parameters['recurrent_0']),
        })
        template = '\t'.join(
            ['%(n_iter)i', '%(time)g', '%(loss)g', '%(val_loss)g', '%(test_loss)g'])
        row = template % info
        print row

        filtered_info = dict(
            (k, v) for k, v in info.items()
            # if (not isinstance(v, (np.ndarray, gp.garray)) or v.size <= 1) and k not in ('args', 'kwargs'))
            if (not isinstance(v, (np.ndarray,)) or v.size <= 1) and k not in ('args', 'kwargs'))

        for key in filtered_info:
            if isinstance(filtered_info[key], np.float32):
                filtered_info[key] = float(filtered_info[key])
        infos.append(filtered_info)

        plot()