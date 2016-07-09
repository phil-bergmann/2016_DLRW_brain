# TODO: move this to a config.ini file
DATA_PATH = 'data/'
IMAGE_PATH = 'images/'
MAT_SUBDIR = 'matfiles/'
P_FILE_REGEX = 'P*.zip'
N_P_FILES = 12

N_EEG_SENSORS = 32
N_EMG_SENSORS = 5

# defines which events to select and their sequence (0: deactivated, 1: First, 2: Second, ...)
SEQ_EMG_TARGETS = [1, 0, 0, 0, 2]
N_EMG_TARGETS = 2

# defines which events to select and their sequence (0: deactivated, 1: First, 2: Second, ...)
SEQ_EEG_TARGETS = [1, 2, 0, 0, 0]
N_EEG_TARGETS = 2


N_EEG_TIMESTEPS = 6200
N_EMG_TIMESTEPS = 50000

STRIDE_LEN = 300