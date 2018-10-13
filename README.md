# 2016-DLRW-brain
The Braintrust

Report in doc/reports/report.pdf and presentation in doc/presentation/main.pdf!

#### EEG-EMG RNN training ####
* Run brain/\_\_main\_\_.py to configure and execute training on EMG and EEG data. 
* breze_RNN.py EMG training implementation
* breze_EEG.py EEG training implementation
* globals.py global static variables
* data preprocessing: data.py (no explicit call necessary)

#### Configuration ####
Overwrite default values calling _RNN_EEG/test_RNN_ method

* n_neurons: number of neurons per hidden layer
* batch_size: number of subsets
* participant: list of experimentees to train on
* series: list of series to train on
* subsample: frequency to subsample EMG data
* imp_weights_skip: number of important weights to skip
* n_layers: number of hidden layers

#### Visualization ####
* data_visualization.py 
* eeg_plotter.py for visualization of raw data
* /img contains single training error plots
* /images contains multiple trial training plots

##### t-SNE #####
* t-SNE call from: bhtsne.py
* tsne.py
* FNN implementation: fnn.py
* C implementation within /windows subdirectory

#### Miscellaneous ####
* data source: https://ndownloader.figshare.com/files/3229301
* data documentation: http://www.nature.com/articles/sdata201447
* data will be downloaded into /data and extracted to /data/matlab by initial execution 
* detailed final report and presentation within /doc subdirectory
* t-SNE examination report within /doc/reports/eeg_curiosities

#### Deprecated ####
* \_\_init\_\_.py
* breze_RNN_old.py
* test_getTables.py (function call test)
* matlab_data.py (data structure analysis)
* elman.py (alternative approach, not continued)
* lstm.py (alternative approach, not continued)
* rnnrbm.py (alternative approach, not continued)
* LogReg.py (alternative approach, not continued)
