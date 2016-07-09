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
from climin.initialize import bound_spectral_radius

import matplotlib.pyplot as plt

def get_shaped_input(participant, series, subsample=0):
    '''

    receive and reshape (emg) data to:
    X = [300 x 2428 x 5]
    Z = [300 x 1320 x 2]

    IMPORTANT: Breze wants the data to be aligned (timesteps, samples, dimensionality)

    :param participant:
    :param series:
    :return:
    '''
    data, eventNames = get_eeg_emg(participant, series, "emg")
    p_train = 0.7
    p_val = 0.15
    n_train = int(len(data) * p_train)
    n_val = int(len(data) * (p_val + p_train)) - int(len(data) * p_train)
    n_test = len(data) - n_train - n_val

    print('[*] Samples Train: %d' %(n_train))
    print('[*] Samples Validation: %d' % (n_val))
    print('[*] Samples Test: %d' % (n_test))

    len_arr = [len(trial['emg_target']) for trial in data]
    max_seqlength = max(len_arr)
    min_seqlength = min(len_arr)
    print('[*] min seqlength: %i' % min_seqlength)

    seqlength = min_seqlength

    X = np.zeros((max_seqlength, n_train, st.N_EMG_SENSORS))
    Z = np.zeros((max_seqlength, n_train, st.N_EMG_TARGETS))
    VX = np.zeros((max_seqlength, n_val, st.N_EMG_SENSORS))
    VZ = np.zeros((max_seqlength, n_val, st.N_EMG_TARGETS))
    TX = np.zeros((max_seqlength, n_test, st.N_EMG_SENSORS))
    TZ = np.zeros((max_seqlength, n_test, st.N_EMG_TARGETS))
    for trial_id in range(len(data)):
        timestep = 0
        for sensor_set in data[trial_id]['emg_target'].iteritems():
            if trial_id < n_train:
                X[timestep, trial_id, ...] = sensor_set[1][0:st.N_EMG_SENSORS]
                Z[timestep, trial_id, ...] = sensor_set[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS+st.N_EMG_TARGETS]
            elif trial_id < n_train + n_val:
                VX[timestep, trial_id-n_train, ...] = sensor_set[1][0:st.N_EMG_SENSORS]
                VZ[timestep, trial_id-n_train, ...] = sensor_set[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS + st.N_EMG_TARGETS]
            else:
                TX[timestep, trial_id - n_train - n_val, ...] = sensor_set[1][0:st.N_EMG_SENSORS]
                TZ[timestep, trial_id - n_train - n_val, ...] = sensor_set[1][st.N_EMG_SENSORS:st.N_EMG_SENSORS + st.N_EMG_TARGETS]
            timestep += 1

    # subsample
    if subsample > 0:
        print('[*] Subsampling with factor %d' %(subsample))

        X = X[::subsample]
        Z = Z[::subsample]

        VX = VX[::subsample]
        VZ = VZ[::subsample]

        TX = TX[::subsample]
        TZ = TZ[::subsample]


        seqlength = seqlength/subsample
        seqlength_mod = seqlength % st.STRIDE_LEN
        seqlength -= seqlength_mod

    print('[*] Seqlenght: ' + str(seqlength))

    # cut data to smallest overlap (along time axis)
    X_trim = X[:seqlength]
    sX = X_trim.transpose(1, 0, 2).reshape((-1, st.STRIDE_LEN, st.N_EMG_SENSORS)).transpose(1, 0, 2)

    Z_trim = Z[:seqlength]
    sZ = Z_trim.transpose(1, 0, 2).reshape((-1, st.STRIDE_LEN, st.N_EMG_TARGETS)).transpose(1, 0, 2)

    VX_trim = VX[:seqlength]
    sVX = VX_trim.transpose(1, 0, 2).reshape((-1, st.STRIDE_LEN, st.N_EMG_SENSORS)).transpose(1, 0, 2)

    VZ_trim = VZ[:seqlength]
    sVZ = VZ_trim.transpose(1, 0, 2).reshape((-1, st.STRIDE_LEN, st.N_EMG_TARGETS)).transpose(1, 0, 2)

    TX = TX[:seqlength]
    TZ = TZ[:seqlength]

    print('[*] Shape Training Set X: ' + str(sX.shape))
    print('[*] Shape Validation Set X: ' + str(sVX.shape))
    print('[*] Shape Test Set X: ' + str(TX.shape))

    return sX, sZ, sVX, sVZ, TX, TZ, seqlength, eventNames


def test_RNN(n_layers = 1, batch_size = 50):
    #optimizer = 'rmsprop', {'step_rate': 0.0001, 'momentum': 0.9, 'decay': 0.9}
    optimizer = 'adadelta', {'decay': 0.9, 'offset': 1e-6, 'momentum': .9, 'step_rate': .1}
    # optimizer = 'adam'
    n_hiddens = [100] * n_layers

    m = SupervisedRnn(
        st.N_EMG_SENSORS, n_hiddens, st.N_EMG_TARGETS,  out_transfer='sigmoid', loss='bern_ces',
        hidden_transfers=['tanh'] * n_layers,
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

    sX, sZ, sVX, sVZ, TX, TZ, seqlength, eventNames = get_shaped_input(1, 1, subsample=10)

    imp_weights_skip = 150
    W = np.ones_like(sZ)
    WV = np.ones_like(sVZ)
    W[:imp_weights_skip, :, :] = 0
    WV[:imp_weights_skip, :, :] = 0

    climin.initialize.randomize_normal(m.parameters.data, 0, 0.1)
    #climin.initialize.bound_spectral_radius(m.parameters.data)

    max_passes = 100
    max_minutes = 10
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
        colors = ['blue', 'red', 'green', 'cyan', 'magenta']
        figure, (axes) = plt.subplots(2, 1)
        x_axis = np.arange(seqlength)

        #input_for_plot = sVX.transpose(1,0,2).reshape((-1, seqlength, st.N_EMG_SENSORS)).transpose(1,0,2)[:, 0:1, :]
        #target_for_plot = sVZ.transpose(1,0,2).reshape((-1, seqlength, st.N_EMG_TARGETS)).transpose(1,0,2)[:, 0:1, :]
        input_for_plot = TX[:, 1:2, :]
        target_for_plot = TZ[:, 1:2, :]
        result = m.predict(input_for_plot)

        for i in range(st.N_EMG_TARGETS):

            axes[0].set_title('TARGETS')
            axes[0].fill_between(x_axis, 0, target_for_plot[:, 0, i], facecolor=colors[i], alpha=0.8,
                                 label=eventNames[st.SEQ_EMG_TARGETS.index(i)])
            #axes[0].plot(x_axis, target_for_plot[:, 0, i])
            axes[1].set_title('RNN')
            axes[1].plot(x_axis, result[:, 0, i], color=colors[i])

        axes[0].legend(loc=0, shadow=True, fontsize='x-small') # loc: 0=best, 1=upper right, 2=upper left

        figure.subplots_adjust(hspace=0.5)
        figure.savefig('test.png')
        plt.close(figure)

    def report(info):
        #print(info)
        return True

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

        plot()

    m.parameters.data[...] = info['best_pars']
    plot()