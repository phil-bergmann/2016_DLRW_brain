import sys
import zipfile
import re
import glob
from util import extract_mat
import globals as st

file_filter_regex = r'HS_P[0-9]_ST.mat'
# file_filter_regex = r'WS_P[0-9]_S[0-9].mat'

def main():
    archive_files = glob.glob(st.P_FILE_REGEX)
    assert len(archive_files) == st.N_P_FILES, ('Number of P archives found: %i expected %i!' % (len(archive_files), static.n_p_files))

    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()

        for f_mat in mat_file_list:
            if re.search(file_filter_regex, repr(f_mat)):
                mat = extract_mat(f_zip, f_mat)

                # file type - HS_P*_ST.mat
                hs = mat['hs']
                print 'EEG  %s, participant:%i, series:%i, records: %i ' % (repr(f_mat), hs['participant'], hs['series'], len(hs['eeg']['sig']))


if __name__ == "__main__":
    sys.exit(main())