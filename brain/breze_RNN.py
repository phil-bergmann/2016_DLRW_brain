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

def get_shaped_input(participant, series):
    '''

    receive and reshape (emg) data to:
    X = [300 x 2428 x 5]
    Z = [300 x 1320 x 2]

    IMPORTANT: Breze wants the data to be aligned (timesteps, samples, dimensionality)

    :param participant:
    :param series:
    :return:
    '''
    data = get_eeg_emg(participant, series)
    p_train = 0.66
    n_train = int(len(data) * p_train)
    n_val = len(data) - int(len(data) * p_train)

    len_arr = [len(trial['emg_target']) for trial in data]
    max_seqlength = max(len_arr)
    min_seqlength = min(len_arr)
    print('min seqlength: %i' % min_seqlength)

    seqlen_mod_300 = min_seqlength % st.STRIDE_LEN
    seqlength = min_seqlength - seqlen_mod_300
    print('seqlength: %i' % seqlength)
    time_win_train = int(np.floor(seqlength * n_train / st.STRIDE_LEN))   # 33000*22/300 = 2420
    print('time_win: %i' % time_win_train)
    time_win_val = int(np.floor(seqlength * n_val / st.STRIDE_LEN))    # 33000*12/300 = 1320
    print('time_win_val: %i' % time_win_val)

    X = np.zeros((max_seqlength, n_train, st.N_EMG_SENSORS))
    Z = np.zeros((max_seqlength, n_train, st.N_EMG_TARGETS))
    VX = np.zeros((max_seqlength, n_val, st.N_EMG_SENSORS))
    VZ = np.zeros((max_seqlength, n_val, st.N_EMG_TARGETS))
    for trial_id in range(len(data)):
        timestep = 0
        for sensor_set in data[trial_id]['emg_target'].iteritems():
            if trial_id < n_train:
                X[timestep, trial_id, ...] = sensor_set[1][0:st.N_EMG_SENSORS]
                Z[timestep, trial_id, ...] = sensor_set[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS+st.N_EMG_TARGETS]
            else:
                VX[timestep, trial_id-n_train, ...] = sensor_set[1][0:st.N_EMG_SENSORS]
                VZ[timestep, trial_id-n_train, ...] = sensor_set[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS + st.N_EMG_TARGETS]
            timestep += 1

    # cut data to smallest overlap (along time axis)
    X_trim = np.zeros((seqlength, n_train, st.N_EMG_SENSORS))
    X_trim = X[:seqlength]
    sX = np.zeros((st.STRIDE_LEN, time_win_train, st.N_EMG_SENSORS))
    sX = X_trim.reshape((st.STRIDE_LEN, time_win_train, st.N_EMG_SENSORS))

    Z_trim = np.zeros((seqlength, n_train, st.N_EMG_TARGETS))
    Z_trim = Z[:seqlength]
    sZ = np.zeros((st.STRIDE_LEN, time_win_train, st.N_EMG_TARGETS))
    sZ = Z_trim.reshape((st.STRIDE_LEN, time_win_train, st.N_EMG_TARGETS))

    VX_trim = np.zeros((seqlength, n_val, st.N_EMG_SENSORS))
    VX_trim = VX[:seqlength]
    print VX_trim.shape

    sVX = np.zeros((st.STRIDE_LEN, time_win_val, st.N_EMG_SENSORS))
    sVX = VX_trim.reshape((st.STRIDE_LEN, time_win_val, st.N_EMG_SENSORS))
    print sVX.shape

    VZ_trim = np.zeros((seqlength, n_val, st.N_EMG_TARGETS))
    VZ_trim = VZ[:seqlength]
    sVZ = np.zeros((st.STRIDE_LEN, time_win_val, st.N_EMG_TARGETS))
    sVZ = VZ_trim.reshape((st.STRIDE_LEN, time_win_val, st.N_EMG_TARGETS))
    return sX, sZ, sVX, sVZ, min_seqlength

def test_RNN(n_layers = 1, batch_size = 50):

    # optimizer = 'rmsprop', {'step_rate': 0.0001, 'momentum': 0.9, 'decay': 0.9}
    optimizer = 'adadelta', {'decay': 0.9, 'offset': 1e-6, 'momentum': .9, 'step_rate': .1}
    # optimizer = 'adam'
    n_hiddens = [100] * n_layers

    m = SupervisedRnn(
        5, n_hiddens, 2,  out_transfer='sigmoid', loss='bern_ces',
        hidden_transfers=['lstm'] * n_layers,
        batch_size=batch_size,
        imp_weight=True,
        optimizer=optimizer)

    m.exprs['true_loss'] = m.exprs['loss']

    '''
    f_loss = m.function(['inpt', 'target'], 'true_loss')

    def test_nll():
        nll = 0
        n_time_steps = 0
        for x, z in zip(tx, tz):
            nll += f_loss(x[:, np.newaxis], z[:, np.newaxis]) * x.shape[0]
            n_time_steps += x.shape[0]
        return nll / n_time_steps
    '''

    sX, sZ, sVX, sVZ, min_seqlength = get_shaped_input(1, 1)

    imp_weights_skip = 5
    W = np.ones_like(sX)[:, :, 0, np.newaxis]
    WV = np.ones_like(sVX)[:, :, 0, np.newaxis]
    W[:imp_weights_skip, :, :] = 0
    WV[:imp_weights_skip, :, :] = 0

    climin.initialize.randomize_normal(m.parameters.data, 0, 0.01)

    max_passes = 100
    max_minutes = 6
    max_iter = max_passes * sX.shape[1] / m.batch_size
    batches_per_pass = int(math.ceil(float(sX.shape[1]) / m.batch_size))
    pause = climin.stops.ModuloNIterations(batches_per_pass * 1)

    stop = climin.stops.Any([
        climin.stops.TimeElapsed(max_minutes * 60),
        # climin.stops.patience('val_loss', 1000, grow_factor=1.1, threshold=0.0001),
        climin.stops.NotBetterThanAfter(30, 100),
    ])

    start = time.time()
    header = '#', 'seconds', 'loss', 'val loss', 'test loss'
    print '\t'.join(header)

    def plot():
        figure, (axes) = plt.subplots(4, 1)
        x_axis = np.arange(min_seqlength)

        result = m.predict(sVX[:, 0:1, :])

        axes[0].set_title("hand_move_target")
        axes[0].plot(x_axis, sVZ[:, 0, 0])
        axes[1].set_title("grasp_target")
        axes[1].plot(x_axis, sVZ[:, 0, 1])
        axes[2].set_title("hand_move_rnn")
        axes[2].plot(x_axis, result[:, 0, 0])
        axes[3].set_title("grasp_rnn")
        axes[3].plot(x_axis, result[:, 0, 1])

        figure.subplots_adjust(hspace=0.5)
        figure.savefig('test.png')
        plt.close(figure)

    def report(info):
        print(info)
        return False

    infos = []
    for i, info in enumerate(m.powerfit((sX, sZ, W), (sVX, sVZ, WV), stop=stop, report=report, eval_train_loss=True)):
        info['loss'] = float(info['loss'])
        # info['val_loss'] = float(info['val_loss'])
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

        m.parameters.data[...] = info['best_pars']
        plot()
