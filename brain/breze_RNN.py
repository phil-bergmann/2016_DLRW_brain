import cPickle
import math
import time

import numpy as np
import theano.tensor as T

from breze.learn.data import interleave, padzeros, split

from breze.learn import base
from breze.learn.rnn import SupervisedFastDropoutRnn, SupervisedRnn
import breze.learn.display as D

import climin.initialize
import climin.stops
import climin.mathadapt as ma

def train_rnn():
    n_layers = 1

    optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9}
    #optimizer = 'adadelta', {'decay': 0.9, 'offset': 1e-6, 'momentum': .9, 'steprate': .1}
    #optimizer = 'gd', {'steprate': 1e-4, 'momentum': .99, 'momentum_type': 'nesterov'}
    #optimizer = 'adamdelta'

    batch_size = 50

    n_hiddens = [100] * n_layers

    from breze.arch.component.varprop.transfer import tanh

    m = SupervisedRnn(
        5, n_hiddens, 2,  out_transfer='sigmoid', loss='bern_ces',
        hidden_transfers=['tanh'] * n_layers,
        batch_size=batch_size,
        optimizer=optimizer)

    m.exprs['true_loss'] = m.exprs['loss']

    f_loss = m.function(['inpt', 'target'], 'true_loss')

    def test_nll():
        nll = 0
        n_time_steps = 0
        for x, z in zip(tx, tz):
            nll += f_loss(x[:, np.newaxis], z[:, np.newaxis]) * x.shape[0]
            n_time_steps += x.shape[0]
        return nll / n_time_steps

    infos = []

    climin.initialize.randomize_normal(m.parameters.data, 0, 0.01)