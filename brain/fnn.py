from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import itertools

from LogReg import LogisticRegression, load_data
#from util import Model, Scores, loadMNIST
from tsne import get_data, get_ws, shuffle
import six.moves.cPickle as pickle
import climin, climin.util, climin.initialize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sys import stdout

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.W_shape = (n_in, n_out)
        self.b_shape = (n_out,)

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out, activation=T.tanh, parameters=None):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation,
            W=parameters[0],
            b=parameters[1]
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=parameters[2],
            b=parameters[3]
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors

        self.params = np.asarray(self.hiddenLayer.params + self.logRegressionLayer.params)

        self.input = input

        self.predict = theano.function(inputs=[input], outputs=self.logRegressionLayer.p_y_given_x)
        self.predict_class = theano.function(inputs=[input], outputs=self.logRegressionLayer.y_pred)




def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=100, n_hidden=300, optimizer='GradientDescent', activation=T.tanh):

    #---- Configure ----
    participant = 1
    series = 1
    datatype = 'eeg'
    trials_from = 1
    trials_to = 'end'
    #-------------------


    # Get data
    ws = get_ws(participant=participant, series=series)
    windows = ws.get('win')
    (data, trials, led) = get_data(windows, datatype=datatype, trials_from=trials_from, trials_to=trials_to)
    (temp, undo_shuffle) = shuffle(np.c_[data, trials - 1])
    n = data.shape[0]
    n_train = 4 * n // 9
    n_valid = 2 * n // 9
    n_test = n - n_train - n_valid

    train_set_x, train_set_y = (temp[:n_train, :data.shape[1]], temp[:n_train, data.shape[1]:])
    valid_set_x, valid_set_y = (temp[n_train:n_train+n_valid, :data.shape[1]], temp[n_train:n_train+n_valid, data.shape[1]:])
    test_set_x, test_set_y = (temp[n_train+n_valid:, :data.shape[1]], temp[n_train+n_valid:, data.shape[1]:])

    # Reshaping data from (n,1) to (n,)
    train_set_y = train_set_y.reshape(train_set_y.shape[0],)
    valid_set_y = valid_set_y.reshape(valid_set_y.shape[0], )
    test_set_y = test_set_y.reshape(test_set_y.shape[0], )



    n_train_batches = train_set_x.shape[0] // batch_size
    print('Building the Model...')

    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    n_in = data.shape[1]
    n_out = 34
    dims = [(n_in, n_hidden), n_hidden, (n_hidden, n_out), n_out]
    flat, (hidden_W, hidden_b, logreg_W, logreg_b) = climin.util.empty_with_views(dims)
    climin.initialize.randomize_normal(flat, loc=0, scale=0.1)
    #hidden_W[...] = np.asarray(rng.uniform(low=-4*np.sqrt(6. / (n_in + n_hidden)), high=4*np.sqrt(6. / (n_in + n_hidden)), size=(n_in, n_hidden)))
    parameters = [theano.shared(value = hidden_W, name = 'W', borrow = True),
                  theano.shared(value=hidden_b, name='b', borrow=True),
                  theano.shared(value = logreg_W, name = 'W', borrow = True),
                  theano.shared(value=logreg_b, name='b', borrow=True)]

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        activation=activation,
        parameters=parameters
    )


    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    gparams = [T.grad(cost, param) for param in classifier.params]


    """ Theano functions """
    grad_W = theano.function([x, y], gparams, allow_input_downcast=True)


    print('Setting up Climin...')
    """ Setting up Climin """

    def d_loss(parameters, inputs, targets):
        g_hl_W, g_hl_b, g_lr_W, g_lr_b = grad_W(inputs, targets)

        return np.concatenate([g_hl_W.flatten(), g_hl_b, g_lr_W.flatten(), g_lr_b])



    minibatch = True
    if not minibatch:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
    else:
        args = ((i, {}) for i in climin.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0]))

    if optimizer=='GradientDescent':
        print('Running GradientDescent')
        opt = climin.GradientDescent(flat, d_loss, step_rate=0.001, momentum=0.95, args=args)
    elif optimizer=='RmsProp':
        print('Running RmsProp')
        opt = climin.rmsprop.RmsProp(flat, d_loss, step_rate=0.001, args=args)
    #elif optimizer == 'NonlinearConjugateGradient':
    #    opt = climin.cg.NonlinearConjugateGradient(d_loss, loss, d_loss, min_grad=1e-06, args=args)
    elif optimizer == 'Adadelta':
        print('Running Adadelta')
        opt = climin.adadelta.Adadelta(flat, d_loss, step_rate=0.01, decay=0.9, momentum=0, offset=0.001, args=args)
    elif optimizer == 'Adam':
        print('Running Adam')
        opt = climin.adam.Adam(flat, d_loss, step_rate=0.001, decay=0.3, decay_mom1=0.1,
                                    decay_mom2=0.001, momentum=0, offset=1e-08, args=args)
    elif optimizer == 'Rprop':
        print('Running Rprop')
        opt = climin.rprop.Rprop(flat, d_loss, step_shrink=0.5, step_grow=1.2, min_step=1e-06, max_step=1,
                                      changes_max=0.1, args=args)
    else:
        print('Optimizer not available!')
        opt = None

    zero_one_loss = theano.function(
        inputs=[x, y],
        outputs=classifier.logRegressionLayer.errors(y),
        allow_input_downcast=True
    )

    print('Running Optimization...\n')

    patience = 10000
    patience_increase = 4
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    train_error_list = []
    valid_error_list = []
    test_error_list = []

    #model = Model(classifier.params)

    train_score = zero_one_loss(train_set_x, train_set_y) * 100
    this_validation_loss = zero_one_loss(valid_set_x, valid_set_y) * 100
    test_score = zero_one_loss(test_set_x, test_set_y) * 100

    train_error_list.append(train_score)
    valid_error_list.append(this_validation_loss)
    test_error_list.append(test_score)


    for info in opt:
        iter = info['n_iter']

        if (iter % 1)==0:
            stdout.write("\r%f%% of Epoch %d" % (float(iter * 100)/n_train_batches - epoch * 100, epoch))
            stdout.flush()

        if (iter + 1) % validation_frequency == 1:
            epoch += 1

            train_score = zero_one_loss(train_set_x, train_set_y) * 100
            this_validation_loss = zero_one_loss(valid_set_x, valid_set_y) * 100
            test_score = zero_one_loss(test_set_x, test_set_y) * 100

            train_error_list.append(train_score)
            valid_error_list.append(this_validation_loss)
            test_error_list.append(test_score)



            print('\nEpoch %i, Validation Error:\t %f%%' % (epoch, this_validation_loss))

            if this_validation_loss < best_validation_loss:
                if (this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_test_score = test_score
                best_iter = iter

                print(('Epoch %i, Test Error:\t %f%% \t NEW MODEL') % (epoch, test_score))
                #with open('model.pkl', 'wb') as f:
                    #print('Dump Model')
                #    pickle.dump(model, f)

            if (epoch >= n_epochs) or done_looping:
                break

            print ('')

        if patience <= iter:
            done_looping = True
            break


    #scores = Scores(train_error_list, valid_error_list, test_error_list, [best_validation_loss, test_score])
    #with open('scores.pkl', 'wb') as f:
    #    pickle.dump(scores, f)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss, best_iter + 1, best_test_score))
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    return (train_error_list, valid_error_list, test_error_list), (best_validation_loss, best_test_score)



def plot_error_curves(error_lists, best_scores, args=('Sigmoid', 'RmsProp')):
    train_error_list, valid_error_list, test_error_list = error_lists
    best_validation_score, best_test_score = best_scores

    fig, curve = plt.subplots(figsize=(20,6))

    curve.plot(train_error_list, '-', linewidth=1, label='Training Error', color='red')
    curve.plot(valid_error_list, '-', linewidth=1, label='Vaidation Error', color='green')
    curve.plot(test_error_list, '-', linewidth=1, label='Test Error', color='blue')
    curve.set_title('Error curves for %s-activated hidden neurons and %s - optimization\nBest model with %.2f%%'
        ' validation error and %.2f%% test error' % (args[0], args[1], best_validation_score, best_test_score))

    fig.canvas.set_window_title('Error curves for a %s-activation function in the hidden neurons' % activation)
    train_plt = mpatches.Patch(color='red', label='Training Error')
    plt.legend(handles=[train_plt])
    valid_plt = mpatches.Patch(color='green', label='Validation Error')
    plt.legend(handles=[train_plt])
    test_plt = mpatches.Patch(color='blue', label='Test Error')
    plt.legend(handles=[train_plt, valid_plt, test_plt])

    plt.ylabel('Error in %')
    plt.xlabel('Epoch')

    plt.show()


if __name__ == '__main__':
    activation = T.nnet.sigmoid
    #activation = T.tanh
    #activation = T.nnet.relu
    optimizer = 'RmsProp'
    lists, best_scores = test_mlp(activation=activation, optimizer=optimizer)
    plot_error_curves(lists, best_scores, args=('Sigmoid', optimizer))
