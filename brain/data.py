# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Roman C. Podolski <roman.podolski@tum.de>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer or coffee in return

"""
TODO Write docstring

"""

from __future__ import print_function

import os
import errno
import zipfile as z

import scipy.io as spio
import scipy
import numpy as np
import globals as st
import glob
import re

# TODO: load the data dir from config.ini
DATA_DIR = 'dataset'

def load_data(participant, type_of_data='eeg'):
    """
    TODO: add detailed documentation

    """

    download_way_eeg_gal(participant)
    unzip_way_eeg_gal(participant)

    if type_of_data ==  'eeg':
        X, Y = load_eeg(participant)
    elif type_of_data == 'emg':
        X, Y = load_emg(participant)
    else:
        raise 'unknown type of data: %s' % type_of_data

    # eeg eeg_t inputs
    # eeg_t inputs

    train_set = (X, Y)
    # TODO: X cross fold validation
    valid_set = ([], []) # currently no x-cross fold validation done!
    test_set = ([], [])
    return train_set, valid_set, test_set

def load_eeg(participant):
    """TODO: Docstring for load_eeg.

    :participant: TODO
    :returns: TODO

    """

    mat_file = os.path.join(DATA_DIR, 'HS_P%d_ST.mat' % participant)
    if(not os.path.isfile(mat_file)):
        unzip_way_eeg_gal(participant)
    data = loadnestedmat(mat_file)['hs']
    print('loading eeg data for participant #%d: %s ' % (
        participant, data['name']
    ))
    eeg_x = data['eeg']['sig']
    # TODO: load targets
    eeg_y = [] # targets currently unknown
    return eeg_x, eeg_y

def load_emg(participant):
    """TODO: Docstring for load_emg.

    :participant: TODO
    :returns: TODO

    """
    pass

def download_way_eeg_gal(participant, dir=DATA_DIR):
    """Downloads the WAY-EEG-GAL dataset

    Downloads the zipped WAY-EEG-GAL dataset of one given participant

    :type participant: int
    :param participant: participant data to download

    :type dir: string
    :param dir: path to dataset folder
    """

    origins = [('P1.zip', 'https://ndownloader.figshare.com/files/3229301'),
               ('P2.zip', 'https://ndownloader.figshare.com/files/3229304'),
               ('P3.zip', 'https://ndownloader.figshare.com/files/3229307'),
               ('P4.zip', 'https://ndownloader.figshare.com/files/3229310'),
               ('P5.zip', 'https://ndownloader.figshare.com/files/3229313'),
               ('P6.zip', 'https://ndownloader.figshare.com/files/3209486'),
               ('P7.zip', 'https://ndownloader.figshare.com/files/3209501'),
               ('P8.zip', 'https://ndownloader.figshare.com/files/3209504'),
               ('P9.zip', 'https://ndownloader.figshare.com/files/3209495'),
               ('P10.zip', 'https://ndownloader.figshare.com/files/3209492'),
               ('P11.zip', 'https://ndownloader.figshare.com/files/3209498'),
               ('P12.zip', 'https://ndownloader.figshare.com/files/3209489')]

    data_file, url = origins[participant - 1]

    try:
        os.makedirs(dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise

    new_path = os.path.join(dir, data_file)

    if(not os.path.isfile(new_path)):
        from six.moves import urllib
        import progressbar as pb

        print('Downloading %s from %s saving to %s ...\n' % (
            data_file, url, new_path
        ))

        widgets = [
            pb.Percentage(), ' ',
            pb.Timer(), ' ',
            pb.Bar(), ' ',
            pb.AnimatedMarker(), ' ',
            pb.ETA()
        ]

        # TODO: clint also offers a progress bar, use the one from clint
        pbar = pb.ProgressBar(widgets=widgets).start()

        def rephook(blocks_transfered, block_size, total_size):
            pbar.maxval = total_size // block_size + 1
            pbar.update(blocks_transfered)

        try:
            urllib.request.urlretrieve(url, new_path, reporthook=rephook)
        except OSError as exc:
            os.remove(new_path)
            return download_way_eeg_gal(participant, dir)

        pbar.finish()


def unzip_way_eeg_gal(participant, dir=DATA_DIR):
    """TODO: Docstring for unzip_way_eeg_gal.

    :participant: TODO

    """
    try:
        new_path = os.path.join(dir, 'P%d.zip' % participant)
        with z.ZipFile(new_path, 'r') as data_zip:
            for fname in data_zip.namelist():
                fpath = os.path.join(dir, fname)

                if(not os.path.isfile(fpath)):
                    print('Extracting %s to %s ...' % (fname, fpath))
                    data_zip.extract(fname, dir)
    except z.BadZipfile:
        os.remove(new_path)
        download_way_eeg_gal(participant, dir)


def loadnestedmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dictt):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dictt:
        if isinstance(dictt[key], spio.matlab.mio5_params.mat_struct):
            dictt[key] = _todict(dictt[key])
    return dictt

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if type(elem).__module__ == np.ndarray.__module__ and any(isinstance(x, scipy.io.matlab.mio5_params.mat_struct) for x in elem):
            elem = [_todict(a) for a in elem]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def extract_mat(zf, filename, relative_path=''):
    '''
    Comment from phil:
    Would seem logic to exctract here only and not to return anything, as this is loadnestedmat() for
    '''
    try:
        zf.extract(filename, path=relative_path+st.MAT_SUBDIR)
        return loadnestedmat(relative_path+st.MAT_SUBDIR+filename)
    except KeyError:
        print('ERROR: Did not find %s in zip file' % filename)

class Regexhandler(object):

    def __init__(self, reg):
        self.regex = reg

    def _extract(self, match):
        if match is not None:
            match = match.regs[0][1]
        end_pattern = re.compile('_|\.mat')
        end = end_pattern.search(self.regex, match)
        if end is not None:
            end = end.regs[0][0]
        return self.regex[match: end]

    def get_series(self):
        try:
            start_pattern = re.compile('_S')
            series_start = start_pattern.search(self.regex)
            return self._extract(series_start)
        except:
            raise Exception('Failed to extract series')

    def get_participant(self):
        try:
            start_pattern = re.compile('_P')
            participant_start = start_pattern.search(self.regex)
            return self._extract(participant_start)
        except:
            raise Exception('Failed to extract participant')

def getTables(regex):
    '''
    A function that returns all Tables matching to a particular regular expression
    Also extracts Tables that aren't yet present in st.MAT_SUBDIR

    e.g. r'WS_P1_S[0-9].mat' should return all windowed session tables as a list from Person 1
    '''

    reghandler = Regexhandler(regex)
    series_regex = reghandler.get_series()
    participant_regex = reghandler.get_participant()

    data = []
    archive_files = glob.glob(st.DATA_PATH + '/' + st.P_FILE_REGEX)
    for archive in archive_files:
        f_zip = z.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()
        for f_mat in mat_file_list:
            if re.search(regex, f_mat):
                mat = extract_mat(f_zip, f_mat)
                ws = mat.get('ws')
                participant = ws.get('participant')
                series = ws.get('series')

                series_pattern = re.compile(series_regex)
                series_target = series_pattern.search(str(series))

                series_pattern = re.compile(participant_regex)
                participant_target = series_pattern.search(str(participant))

                if (series_target is not None) and (participant_target is not None):
                    ws = mat.get('ws')
                    for win in enumerate(ws.get('win')):
                        data.append(win[1])
    return data

def getRaw(regex):
    '''
    A function that returns all Tables matching to a particular regular expression
    Also extracts Tables that aren't yet present in st.MAT_SUBDIR

    e.g. r'WS_P1_S[0-9].mat' should return all windowed session tables as a list from Person 1

    This function returns just the raw tables and does not do any preprocessing
    '''


    data = []
    archive_files = glob.glob(st.DATA_PATH + st.P_FILE_REGEX)
    for archive in archive_files:
        f_zip = z.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()
        for f_mat in mat_file_list:
            if re.search(regex, f_mat):
                mat = extract_mat(f_zip, f_mat)
                data.append(mat)
    return data

def get_emg(participant, series):
    '''
    returns a list of dicts, having event times and dicts of eeg and emg data

    data (eeg/emg) dicts keys: windowed time in seconds from eeg_t/emg_t tables

    eeg data format: 32 channels + 1 target
    emg data format: 5 channels + 1 target

    target currently: hand move window - calcuated from AllLifts events

    :param participant e.g. 1, [0-9]
    :param series e.g. 1, [0-9]

    :return: list of dicts of eeg, emg data
    '''
    allTrials = []
    allLifts = getRaw(r'P'+str(participant)+'_AllLifts.mat')[0].get('P')
    allLifts_colNames = allLifts.get('ColNames')
    data_allLifts = allLifts.get('AllLifts')
    for sample in data_allLifts:
        allTrials.append(dict(zip(np.asarray(allLifts_colNames), sample)))

    data = []
    ws_data = getTables(r'WS_P' + str(participant) + '_S' + str(series) + '.mat')
    for trial_id, win in enumerate(ws_data):
        trial_tHandStart = allTrials[trial_id].get('tHandStart')
        trial_DurReach = allTrials[trial_id].get('Dur_Reach')
        trial_DurPreload = allTrials[trial_id].get('Dur_Preload')

        eeg_data = np.zeros((6000, st.N_EEG_SENSORS + st.N_TARGETS))
        e = win.get('eeg')
        eeg_data[:e.shape[0],0:st.N_EEG_SENSORS] = e
        eeg_t = win.get('eeg_t')
        eeg_dict = dict(zip(eeg_t, eeg_data))

        for item in eeg_dict.iteritems():
            key = item[0]
            item[1][st.N_EEG_SENSORS] = key > trial_tHandStart and key < trial_tHandStart+trial_DurReach
        # eeg_target_vec = {key: key > trial_tHandStart and key < trial_tHandStart+trial_DurReach for key in eeg_dict.iterkeys()}

        emg_data = np.zeros((50000, st.N_EMG_SENSORS + st.N_TARGETS))
        e = win.get('emg')
        emg_data[:e.shape[0],0:st.N_EMG_SENSORS] = e
        emg_t = win.get('emg_t')
        emg_dict = dict(zip(emg_t, emg_data))

        for item in emg_dict.iteritems():
            key = item[0]
            item[1][st.N_EMG_SENSORS] = key > trial_tHandStart and key < trial_tHandStart+trial_DurReach

        data.append({'trial_id': trial_id,'eeg': eeg_dict, 'emg': emg_dict,
                     'tHandStart': trial_tHandStart, 'DurReach': trial_DurReach, 'Dur_Preload':trial_DurPreload})

    return data

def normalize(data):
    '''
    abs(data)
    max
    data / max
    data*2

    :param data:

    :return: normalized data
    '''

    return None
