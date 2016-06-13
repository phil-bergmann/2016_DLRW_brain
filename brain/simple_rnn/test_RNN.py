from __future__ import print_function
import six.moves.cPickle as pickle

import numpy
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util

import brain.data.globals as st
from brain.data.util import getTables, getRaw

from RNN import RNN

WS_file_filter_regex = r'WS_P[0-9]*_S[0-9].mat'
WS_file_filter_regex_P1 = r'WS_P1_S[0-9].mat'
AllLifts_P1 = r'P1_AllLifts.mat'

def test_RNN(nh=300, nl=3, n_in=32):
    # nearest element in list min(myList, key=lambda x:abs(x-myNumber))
    # Wait until implemented in brain.data.util
    data = getTables(WS_file_filter_regex_P1)
    datasetSize = len(data)

    # HandStart(33), LiftOff(18)
    event_data = getRaw(AllLifts_P1)[0]['P']['AllLifts']

    handStart = event_data[:, 33]
    liftOff = event_data[:, 18]

    # get nearest index in eeg data for given event and get length of longest eeg window
    handStartIndex = []
    liftOffIndex = []
    maxLengthEEG = 0
    for i in range(len(data)):
        handStartIndex.append(numpy.where(data[i]['eeg_t'] == min(data[i]['eeg_t'], key=lambda x:abs(handStart[i]-x)))[0][0])
        liftOffIndex.append(numpy.where(data[i]['eeg_t'] == min(data[i]['eeg_t'], key=lambda x:abs(liftOff[i]-x)))[0][0])
        if len(data[i]['eeg']) > maxLengthEEG:
            maxLengthEEG = len(data[i]['eeg'])

    # Construct target vectors (0 = 'none' event)
    targets = numpy.zeros((datasetSize, maxLengthEEG), dtype='int64')
    for i in range(datasetSize):
        targets[i][handStartIndex[i]] = 1
        targets[i][liftOffIndex[i]] = 2

    # Construct data array with 0 padding at the end for shorter sequences
    eeg_data = numpy.zeros((datasetSize, maxLengthEEG, 32))
    for i in range(datasetSize):
        eeg_data[i, 0:data[i]['eeg'].shape[0]] = data[i]['eeg']


    tmpl = [(n_in, nh), (nh, nh), (nh, nl), nh, nl, nh]
    wrt, (Wx, Wh, W, bh, b, h0) = climin.util.empty_with_views(tmpl)
    params = [Wx, Wh, W, bh, b, h0]

    x = T.dmatrix('x')
    y = T.lvector('y')

    classifier = RNN(x, y, nh, nl, n_in)

    # copy preinitialized weight matrices
    Wx[...] = classifier.Wx.get_value(borrow=True)[...]
    Wh[...] = classifier.Wh.get_value(borrow=True)[...]
    W[...] = classifier.W.get_value(borrow=True)[...]


    def set_pars():
        for p, p_class in zip(params, classifier.params):
            p_class.set_value(p, borrow=True)

    def loss(parameters, inpt, targets):
        set_pars()
        return classifier.cost.eval({x: inpt[0], y: targets[0]})

    def d_loss_wrt_pars(parameters, inpt, targets):
        set_pars()
        grads = []
        print(loss(parameters, inpt, targets))
        for d in classifier.gradients:
            grads.append(d.eval({x: inpt[0], y: targets[0]}))
        return numpy.concatenate([grads[0].flatten(), grads[1].flatten(), grads[2].flatten(), grads[3], grads[4], grads[5]])


    args = ((i, {}) for i in climin.util.iter_minibatches([eeg_data[0:1], targets[0:1]], 1, [0, 0]))

    opt = climin.adadelta.Adadelta(wrt, d_loss_wrt_pars, step_rate=1, decay=0.9, momentum=0, offset=0.0001, args=args)

    for info in opt:
        iteration = info['n_iter']
        if iteration > 10:
            break


