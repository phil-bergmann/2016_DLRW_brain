from __future__ import print_function
from __future__ import absolute_import
#import six.moves.cPickle as pickle

import numpy
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util
import matplotlib.pyplot as plt

from data import load_data
from data import getTables
from data import getRaw

#import brain.data.globals as st
#from data.util import getTables, getRaw

class RNN(object):
    """ A simple RNN

    A first try to implement a simple RNN, mostly for the EEG Tasks.
    Based upon elman.py and rnnslu.py in this folder.
    """

    def __init__(self, input, output, nh, nl, n_in=32):
        """ Initialize all parameters for the RNN

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (timesteps, n_in)

        :type output: theano.tensor.dmatrix
        :param output: a symbolic tensor of shape (timesteps) containing the index of class we expect, don't forget to
        add a "none" class for every timestep nothing is supposed to happen

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nl: int
        :param nl: number of labels to predict, don't forget to account for a "none" label

        :type n_in: int
        :param n_in: dimension of the input
        """

        # parameters of the model
        self.Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_in, nh)).astype(theano.config.floatX))
        self.Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        self.W = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nl)).astype(theano.config.floatX))
        self.bh = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b = theano.shared(numpy.zeros(nl, dtype=theano.config.floatX))
        self.h0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        self.x = input
        self.y = output

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, sequences=self.x, outputs_info=[self.h0, None], n_steps=self.x.shape[0])

        self.result_sequence = s[:, 0, :]

        # cost is defined as difference between desired output and the result sequence
        nll = -T.mean(T.log(self.result_sequence)[T.arange(output.shape[0]), output])
        self.gradients = T.grad(nll, self.params)

        self.cost = nll

        #updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, gradients))

def test_RNN(nh=100, nl=3, n_in=32):

    WS_file_filter_regex = r'WS_P[0-9]*_S[0-9].mat'
    WS_file_filter_regex_P1 = r'WS_P1_S[0-9].mat'
    AllLifts_P1 = r'P1_AllLifts.mat'
    # nearest element in list min(myList, key=lambda x:abs(x-myNumber))
    # Wait until implemented in brain.data.util

    # needed are eeg and eeg_t data as train set data
    # for now train and test set contain empty lists
    train_set, valid_set, test_set = load_data(participant=1)

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y   = test_set

    data = getTables(WS_file_filter_regex_P1)
    datasetSize = len(data)

    print(data)
    print(datasetSize)

    # HandStart(33), LiftOff(18)
    # returns list targets
    event_data = getRaw(AllLifts_P1)[0]['P']['AllLifts']

    handStart = event_data[:, 33]
    liftOff = event_data[:, 18]

    # get nearest index in eeg data for given event and get length of longest/shortest eeg window
    handStartIndex = []
    liftOffIndex = []
    maxLengthEEG = 0
    minLengthEEG = numpy.inf
    for i in range(len(data)):
        handStartIndex.append(numpy.where(data[i]['eeg_t'] == min(data[i]['eeg_t'], key=lambda x: abs(handStart[i]-x)))[0][0])
        liftOffIndex.append(numpy.where(data[i]['eeg_t'] == min(data[i]['eeg_t'], key=lambda x: abs(liftOff[i]-x)))[0][0])
        if len(data[i]['eeg']) > maxLengthEEG:
            maxLengthEEG = len(data[i]['eeg'])
        if len(data[i]['eeg']) < minLengthEEG:
            minLengthEEG = len(data[i]['eeg'])

    sequenceLength = maxLengthEEG

    # Construct target vectors (0 = 'none' event)
    targets = numpy.zeros((datasetSize, sequenceLength), dtype='int64')
    for i in range(datasetSize):
        targets[i][handStartIndex[i]-100: handStartIndex[i]+100] = 1
        targets[i][liftOffIndex[i]-100: liftOffIndex[i]+100] = 2
        #print(str(handStartIndex[i]) + " " + str(liftOffIndex[i]))

    # Construct data array with 0 padding at the end for shorter sequences
    eeg_data = numpy.zeros((datasetSize, sequenceLength, 32))
    for i in range(datasetSize):
        eeg_data[i, 0:data[i]['eeg'].shape[0]] = data[i]['eeg']
        #eeg_data[i, :] = data[i]['eeg'][0: sequenceLength]

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

    def plot():
        figure, (axes) = plt.subplots(4, 1)

        x_axis = numpy.arange(sequenceLength)

        result = classifier.result_sequence.eval({x: eeg_data[0]})

        axes[0].set_title("labels")
        axes[0].plot(x_axis, targets[0], label="targets")
        axes[1].set_title("none_prob")
        axes[1].plot(x_axis, result[:, 0], label="none")
        axes[2].set_title("handStart_prob")
        axes[2].plot(x_axis, result[:, 1], label="handStart")
        axes[3].set_title("liftOff_prob")
        axes[3].plot(x_axis, result[:, 2], label="liftOff")

        figure.subplots_adjust(hspace=0.5)

        figure.savefig('test.png')

        plt.close(figure)

    for info in opt:
        iteration = info['n_iter']
        if iteration % 10 == 0:
            plot()
        if iteration > 500:
            break

    plot()
