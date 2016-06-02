import sys
import static as st
import zipfile
import re
import glob
from eeg_plotter import *
from matlab_utils import extract_mat, _check_keys, _todict

HS_file_filter_regex = r'HS_P[0-9]_ST.mat'
WS_file_filter_regex = r'WS_P[0-9]_S[0-9].mat'

def get_time_series_index(series_times_list, event):
    series_idx, = np.where(series_times_list == event)
    while not series_idx and event > 0:
        series_idx, = np.where(series_times_list == event)
        event -= 0.001
    return series_idx

def main():
    archive_files = glob.glob('../'+st.p_file_path)
    assert len(archive_files) == st.n_p_files, ('Number of P archives found %i != %i' % (len(archive_files), st.n_p_files))

    for archive in archive_files:
        f_zip = zipfile.ZipFile(archive, 'r')
        mat_file_list = f_zip.namelist()

        for f_mat in mat_file_list:
            # sys.stdout.write('\r eeg: participant: %s series: %s sensor: %i' % (participant_id, series_id))
            # sys.stdout.flush()

            if re.search(WS_file_filter_regex, repr(f_mat)):
                mat = extract_mat(f_zip, f_mat, relative_path='../')
                ws = mat.get('ws')
                participant_id = ws.get('participant')
                series_id = ws.get('series')
                names = ws.get('names')
                names_eeg = names.get('eeg')
                trial = 0
                windows = ws.get('win')[trial]
                # for trial_id, win in enumerate(windows):
                win = _todict(windows)
                times_eeg = win.get('eeg_t')
                data_eeg = win.get('eeg').transpose()
                ev_led_on = win.get('LEDon')
                ev_led_off = win.get('LEDoff')
                led_on_idx = get_time_series_index(times_eeg, ev_led_on)
                led_off_idx = get_time_series_index(times_eeg, ev_led_off)
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
    os.system('espeak "Visualization has finished"')