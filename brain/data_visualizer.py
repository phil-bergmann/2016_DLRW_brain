import zipfile
import re
import sys
import glob
from eeg_plotter import *
import brain.data.globals as st
from brain.data.util import extract_mat

HS_file_filter_regex = r'HS_P[0-9]_ST.mat'
WS_file_filter_regex = r'WS_P[0-9]_S[0-9].mat'

def time_to_series_index(series_times_list, event_time):
    series_idx, = np.where(series_times_list == event_time)

    # cut off decimal precision, in case of mismatch
    # of event_time in ws.win.eeg_t time lookup table
    time_str = str(event_time)[:-1]
    while not series_idx and event_time > 0:
        f = float(time_str)
        series_idx, = np.where(series_times_list == f)
        time_str = str(time_str)[:-1]

    return series_idx


def main():
    archive_files = glob.glob('../' + st.DATA_PATH + st.P_FILE_REGEX)
    assert len(archive_files) == st.N_P_FILES, ('Number of P archives found %i != %i' % (len(archive_files), st.N_P_FILES))

    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()

        for f_mat in mat_file_list:
            if re.search(WS_file_filter_regex, repr(f_mat)):
                mat = extract_mat(f_zip, f_mat, relative_path='../'+st.DATA_PATH)
                ws = mat.get('ws')
                participant_id = ws.get('participant')
                series_id = ws.get('series')
                names = ws.get('names')
                names_eeg = names.get('eeg')

                windows = ws.get('win')
                for trial, win in enumerate(windows):
                # trial = 5
                # win = ws.get('win')[trial]

                    times_eeg = win.get('eeg_t')
                    data_eeg = win.get('eeg').transpose()

                    led_on_time = win.get('LEDon')
                    led_off_time = win.get('LEDoff')
                    led_on_idx = time_to_series_index(times_eeg, led_on_time)
                    led_off_idx = time_to_series_index(times_eeg, led_off_time)

                    visualize_ws(data_eeg, names_eeg, series_id, participant_id, trial, led_on_idx, led_off_idx)

            # if re.search(HS_file_filter_regex, repr(f_mat)):
            #     mat = extract_mat(f_zip, f_mat, relative_path='../')
            #
            #     # file type - HS_P*_ST.mat
            #     hs = mat.get('hs')
            #     sig = hs.get('eeg').get('sig')
            #     participant = hs.get('participant')
            #     series_id = hs.get('series')
            #     visualize_hs(sig, series_id, hs.get('participant'))
            #     print 'EEG  %s, participant: %i, series:%i, records: %i ' % (repr(f_mat), participant, series_id, len(sig))


if __name__ == "__main__":
    sys.exit(main())
