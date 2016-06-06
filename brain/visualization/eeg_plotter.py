import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import static as st

def visualize_hs(data_dict, series_id, participant_id):
    plt.figure(figsize=(25, 41))
    plt.ioff()

    for win in data_dict:
        name = win[0]
        data = win[1]

        n_sensors = len(data)
        for sensor_id in range(n_sensors):

            ax = plt.subplot2grid((n_sensors, 1), (sensor_id, 0))
            measurement = data#[:,sensor_id]
            ax.plot(range(len(measurement)), measurement)
            mini = np.min(measurement)
            maxi = np.max(measurement)
            step_size = int((maxi-mini)/5)
            ax.set_yticks(xrange(mini, maxi, step_size))
            patch = mpatches.Patch(color='green', label='Sensor '+name)
            ax.legend(handles=[patch], prop={'size':12})

    plt.xlim((0, data.shape[0]))
    plt.xlabel('time')
    plt.tight_layout()
    plt.suptitle('EEG: participant: %s series: %s sensor: %s' % (series_id, sensor_id, participant_id), y=0.0, fontsize=24)
    plt.savefig(st.vis_path_images+'P'+str(participant_id)+'_S'+str(series_id)+'_'+str(sensor_id)+'_'+'.png')
    plt.close()

def visualize_ws(data_eeg, names_eeg, series_id, trial, participant_id, led_on, led_off):
    plt.ioff()
    plt.figure(figsize=(25, 41))
    plt.xlim((0, data_eeg.shape[0]))


    n_samples = len(data_eeg[0])
    n_sensors = len(names_eeg)
    for sensor_id in range(n_sensors):

        ax = plt.subplot2grid((n_sensors, 1), (sensor_id, 0))
        measurement = data_eeg[sensor_id]
        min_y = int(np.min(measurement))
        max_y = int(np.max(measurement))
        step_size = int((max_y - min_y) / 5)
        ax.set_yticks(xrange(min_y, max_y, step_size))
        ax.set_xticks(xrange(0, n_samples, 500))
        ax.plot(range(len(measurement)), measurement)
        ax.autoscale(False)
        patch_eeg = mpatches.Patch(color='green', label=names_eeg[sensor_id])
        ax.legend(handles=[patch_eeg], prop={'size': 12})
        y_min = ax.get_ylim()[0]
        y_max = ax.get_ylim()[1]
        if led_on and led_off:
            r = matplotlib.patches.Rectangle((led_on, y_min), led_off-led_on, y_max*100, fill=True, color='yellow')
            r.set_alpha(0.1)
            ax.add_artist(r)

    print 'participant: %s series: %s trial: %s' % (participant_id, series_id, trial)

    # plt.xlabel('time')
    plt.tight_layout()
    plt.suptitle('EEG: participant: %s series: %s trial: %s' % (participant_id, series_id, trial), y=0.0,
                 fontsize=20)
    plt.savefig(st.vis_path_images+'P' + str(participant_id) + '_S' + str(series_id)+ '_t'+str(trial)+ '.png')
    plt.close()