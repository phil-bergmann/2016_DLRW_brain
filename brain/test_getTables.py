from __future__ import print_function
from data import getTables, getRaw, get_eeg_emg

if __name__ == "__main__":

    emg_tables = get_eeg_emg(participant=11, series=1)
    first_trial = emg_tables[0]
    print('trial id: %i, samples eeg: %i, samples emg: %i, tHandStart: %f, Dur_Reach: %f, Dur_Preload: %f' %
          (first_trial['trial_id'], len(first_trial['eeg_target']), len(first_trial['emg_target']),
           first_trial['tHandStart'], first_trial['DurReach'], first_trial['Dur_Preload']))

    tGrasp_start = first_trial['tHandStart'] + first_trial['DurReach']
    tGrasp_end = tGrasp_start + first_trial['Dur_Preload']
    print('intention to grasp [%f, %f]' % (tGrasp_start, tGrasp_end))

    print('hand moves [%f, %f]' % (first_trial['tHandStart'], first_trial['tHandStart'] + first_trial['DurReach']))
    for rec in first_trial['eeg_target'].iteritems():
        if rec[1][len(rec[1])-1] > 0:
            print(rec[0])


    # table = getTables(r'WS_P11_S[0-9].mat')
    # print len(table)
    # table = getTables(r'WS_P[0-9]_S1.mat')
    # print len(table)
