#!/usr/bin/env /home/dominik/anaconda2/bin/python

import sys
import zipfile
import re
import glob
from matlab_utils import extract_mat
import static

archive_files = glob.glob(static.p_file_path)

file_filter_regex = r'HS_P[0-9]_ST.mat'

def main():
    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()

        for f_mat in mat_file_list:
            if re.search(file_filter_regex, repr(f_mat)):
                mat = extract_mat(f_zip, f_mat)
                hs = mat['hs']
                print 'EEG  %s, participant:%i, series:%i, records: %i ' % (repr(f_mat), hs['participant'], hs['series'], len(hs['eeg']['sig']))


if __name__ == "__main__":
    sys.exit(main())