from __future__ import print_function
import six.moves.cPickle as pickle

import numpy
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util
import glob
import re
import zipfile
import os

import brain.data.globals as st
from brain.data.util import extract_mat, loadnestedmat

from RNN import RNN

WS_file_filter_regex = r'WS_P[0-9]*_S[0-9].mat'
WS_file_filter_regex_P1 = r'WS_P1_S[0-9].mat'

def test_RNN(nh=300, nl=4, n_in=32):
    # Wait until implemented in brain.data.util
    # data = getTable(WS_file_filter_regex_P1)


    tmpl = [(n_in, nh), (nh, nh), (nh, nl), nh, nl, nh]
    flat, (Wx, Wh, W, bh, b, h0) = climin.util.empty_with_views(tmpl)
    params = [Wx, Wh, W, bh, b, h0]

    x = T.lmatrix('x')
    y = T.lmatrix('y')

    classifier = RNN(x, y, nh, nl, n_in)


    def set_pars():
        for p, p_class in zip(params, classifier.params):
            p_class.setValue(p, borrow=True)
