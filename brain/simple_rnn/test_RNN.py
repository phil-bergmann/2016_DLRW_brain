from __future__ import print_function
import six.moves.cPickle as pickle

import numpy
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util

from RNN import RNN

def test_RNN(nh=300, nl=4, n_in=32):

    tmpl = [(n_in, nh), (nh, nh), (nh, nl), nh, nl, nh]  # w is matrix and b a vector
    flat, (Wx, Wh, W, bh, b, h0) = climin.util.empty_with_views(tmpl)
    params = [Wx, Wh, W, bh, b, h0]

    x = T.lmatrix('x')
    y = T.lmatrix('y')

    classifier = RNN(x, y, nh, nl, n_in)


    def set_pars():
        for p, p_class in zip(params, classifier.params):
            p_class.setValue(p, borrow=True)

            


if __name__ == '__main__':
    test_RNN()