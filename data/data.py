#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Roman C. Podolski <roman.podolski@tum.de>
#
# Distributed under terms of the MIT license.


"""
TODO: Write docstring

"""

from __future__ import print_function

import progressbar

import os
import sys

def load_way_eeg_gal(shared = True):
    """Loads the WAY-EEG-GAL dataset

    If the WAY-EEG-GAL dataset is not present in the ./data/ directory
    this will download the zip files for the dataset from figshare.com.

    Unzips the dataset and returns the data in a numpy array

    :type shared:   boolean
    :param dataset: move the data to a shared theano tensor, use a numpy array otherwise

    :returns: the WAY-EEG-GAL dataset

    WARNING!!!!! THE WAY-EEG-GAL dataset is about 15 GB...
    """


    origins = [
            ('P1.zip' , 'https://ndownloader.figshare.com/files/3229301'),
            ('P2.zip' , 'https://ndownloader.figshare.com/files/3229304'),
            ('P3.zip' , 'https://ndownloader.figshare.com/files/3229307'),
            ('P4.zip' , 'https://ndownloader.figshare.com/files/3229310'),
            ('P5.zip' , 'https://ndownloader.figshare.com/files/3229313'),
            ('P6.zip' , 'https://ndownloader.figshare.com/files/3209486'),
            ('P7.zip' , 'https://ndownloader.figshare.com/files/3209501'),
            ('P8.zip' , 'https://ndownloader.figshare.com/files/3209504'),
            ('P9.zip' , 'https://ndownloader.figshare.com/files/3209495'),
            ('P10.zip', 'https://ndownloader.figshare.com/files/3209492'),
            ('P11.zip', 'https://ndownloader.figshare.com/files/3209498'),
            ('P12.zip', 'https://ndownloader.figshare.com/files/3209489') 
            ]

    for [data_file, url] in origins:
        new_path = os.path.join(os.path.split(__file__)[0], data_file)

        if(not os.path.isfile(data_file)): 
            from six.moves import urllib

            pbar = progressbar.ProgressBar().start() 
            def rephook(blocks_transfered, block_size, total_size):
                pbar.maxval = total_size // block_size + 1
                pbar.update(blocks_transfered)
                # print('%d bytes / %d bytes, blocks %d, block size %d' % (blocks_transfered * block_size, total_size, blocks_transfered, block_size))

            print('Downloading %s from %s ...' % (data_file, url))
            urllib.request.urlretrieve(url, data_file, reporthook = rephook)

            pbar.finish()

if __name__ == "__main__":
    load_way_eeg_gal()
