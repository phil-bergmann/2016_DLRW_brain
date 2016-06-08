import scipy.io as spio
import scipy
import numpy as np
import globals as st
import glob
import zipfile
import re
import os

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
        print 'ERROR: Did not find %s in zip file' % filename


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
            # if series_start is not None:
            #     series_start = series_start.regs[0][1]
            # end_pattern = re.compile('_|\.mat')
            # series_end = end_pattern.search(self.regex, series_start)
            # if series_end is not None:
            #     series_end = series_end.regs[0][0]
            return self._extract(series_start)
        except:
            raise Exception('Failed to extract series')
        raise Exception('No series specified')

    def get_participant(self):
        try:
            start_pattern = re.compile('_P')
            participant_start = start_pattern.search(self.regex)
            # if participant_start is not None:
            #     participant_start = participant_start.regs[0][1]
            # end_pattern = re.compile('_|\.mat')
            # participant_end = end_pattern.search(self.regex, participant_start)
            # if participant_end is not None:
            #     participant_end = participant_end.regs[0][0]
            # return self.regex[participant_start : participant_end]
            return self._extract(participant_start)
        except:
            raise Exception('Failed to extract participant')
        raise Exception('No participant specified')

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
    archive_files = glob.glob('../'+st.DATA_PATH + st.P_FILE_REGEX)
    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
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