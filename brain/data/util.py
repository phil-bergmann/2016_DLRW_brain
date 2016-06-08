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

def getTables(regex):
    '''
    A function that returns all Tables matching to a particular regular expression
    Also extracts Tables that aren't yet present in st.MAT_SUBDIR
    
    e.g. r'WS_P1_S[0-9].mat' should return all windowed session tables as a list from Person 1
    '''
    data = []
    archive_files = glob.glob(st.DATA_PATH + st.P_FILE_REGEX)
    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()
        for f_mat in mat_file_list:
            if re.search(regex, f_mat):
                if (not os.path.isfile(st.MAT_SUBDIR + f_mat)):
                    extract_mat(f_zip, f_mat)
                data.append(loadnestedmat(st.MAT_SUBDIR + f_mat))
    return data