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
    data = []
    archive_files = glob.glob(st.DATA_PATH + st.P_FILE_REGEX)
    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()
        for f_mat in mat_file_list:
            if re.search(WS_file_filter_regex_P1, f_mat):
                if (not os.path.isfile(st.MAT_SUBDIR + f_mat)):
                    extract_mat(f_zip, f_mat)
                data.append(loadnestedmat(st.MAT_SUBDIR + f_mat))

    return data


    tmpl = [(n_in, nh), (nh, nh), (nh, nl), nh, nl, nh]
    flat, (Wx, Wh, W, bh, b, h0) = climin.util.empty_with_views(tmpl)
    params = [Wx, Wh, W, bh, b, h0]

    x = T.lmatrix('x')
    y = T.lmatrix('y')

    classifier = RNN(x, y, nh, nl, n_in)


    def set_pars():
        for p, p_class in zip(params, classifier.params):
            p_class.setValue(p, borrow=True)
