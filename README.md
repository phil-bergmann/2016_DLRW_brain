# 2016-DLRW-brain
The Braintrust

#### Run EEG-EMG ####
Run brain/\_\_main\_\_.py to configure and execute training on EMG and EEG data. 
- EMG training implementation: breze_EMG.py
- EEG training implementation: breze_EEG.py

#### t-SNE ####
- t-SNE call from: bhtsne.py
- tsne.py
- FNN implementation: fnn.py
- C implementation within /windows subdirectory

#### Visualization ####
- data_visualization.py 
- eeg_plotter.py for visualization of raw data
- /img contains single training error plots
- /images contains multiple trial training plots

#### Miscellaneous ####
- global static configurations for breze RNN training: globals.py
- data preprocessing: data.py (no explicit call necessary)
- detailed report and presentation within /doc 

#### Deprecated ####
- breze_RNN_old.py
- test_getTables.py (function call test)
- matlab_dat.py (data structure analysis)
- elman.py (alternative approach, not continued)
- lstm.py (alternative approach, not continued)
- rnnrbm.py (alternative approach, not continued)
- LogReg.py (alternative approach, not continued)