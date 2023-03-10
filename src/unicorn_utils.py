import copy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


def makeBigList(lst):
    biglst = []
    for item in lst :
        biglst = biglst + [k for k in range((item-1)*10,item*10)]
    return biglst

def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2, state_space=True):
    filters = []
    for f in range(freq_lim.shape[0]):
        A, B, C, D, Xnn  = init_filt_coef_statespace(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([A, B, C, D, Xnn ])
    return filters

def init_filt_coef_statespace(cuttoff, fs, filtype, order,len_selected_electrodes=27):
    if filtype == 'lowpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    elif filtype == 'bandpass':
        b,a = signal.butter(order, [cuttoff[0]/(fs/2), cuttoff[1]/(fs/2)], filtype)
    elif filtype == 'highpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    # getting matrices for the state-space form of the filter from scipy and init state vector
    A, B, C, D = signal.tf2ss(b,a)
    Xnn = np.zeros((1,len_selected_electrodes,A.shape[0],1))
    return A, B, C, D, Xnn 

def apply_filter_statespace(sig, A, B, C, D, Xnn):
    # State space with scipy's matrices
    filt_sig = np.array([])
    for sample in sig: 
        filt_sig = np.append(filt_sig, C@Xnn + D * sample)
        Xnn = A@Xnn + B * sample
    return filt_sig, Xnn

noASR = 0
total_seg = 0
def pre_processing(curr_segment,selected_electrodes_names ,filters, sample_duration, freq_limits_names, #pipeline_type, 
                    sampling_frequency=250, asr=None):
    outlier = 0
    global noASR, total_seg
    total_seg +=1
    # 1. notch filt
    b_notch, a_notch = signal.iirnotch(50, 30, sampling_frequency)
    for column in curr_segment.columns:
        curr_segment.loc[:,column] = signal.filtfilt(b_notch, a_notch, curr_segment.loc[:,column])
    
    curr_segment = curr_segment.T

#     # 2. Artifact Subspace Reconstruction
#     if asr != None:
#         try:
#             #print('applying ASR...')
#             curr_segment = asr.transform(curr_segment.to_numpy())
#             curr_segment = pd.DataFrame(curr_segment.T, columns = selected_electrodes_names)
#             curr_segment = curr_segment.T
#             #print(segment_filt.shape)
#         except:
#             noASR += 1
#             #print(segment_filt.shape)
#             print(f'didnt apply ASR for {noASR} times out of total of {total_seg} segments.')

    # 3 OUTLIER DETECTION --> https://www.mdpi.com/1999-5903/13/5/103/html#B34-futureinternet-13-00103
    for i, j in curr_segment.iterrows():
        if stats.kurtosis(j) > 4*np.std(j) or (abs(j - np.mean(j)) > 125).any():
            if stats.kurtosis(j) > 4*np.std(j):
                print('due to kurtosis')
            outlier +=1

    # 4 APPLY COMMON AVERAGE REFERENCE (CAR) per segment only for deep learning pipeline   
    # CAR doesnt work for csp, for riemann gives worse results, so only use for deep
    #if 'deep' in pipeline_type: 
    curr_segment -= curr_segment.mean()

    # 5 FILTERING filter bank / bandpass
    segment_filt, filters = filter_1seg_statespace(curr_segment, selected_electrodes_names, filters, sample_duration, 
    freq_limits_names)

    return segment_filt, outlier, filters

def filter_1seg_statespace(segment, selected_electrodes_names,filters, sample_duration, freq_limits_names):
    # filters dataframe with 1 segment of 1 sec for all given filters
    # returns a dataframe with columns as electrode-filters
    filter_results = {}
    segment = segment.transpose()
    for electrode in range(len(selected_electrodes_names)):
        for f in range(len(filters)):
            A, B, C, D, Xnn = filters[f] 
            filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]] = []
            if segment.shape[0] == sample_duration:      
                # apply filter ??nd update Xnn state vector       
                filt_result_temp, Xnn[0,electrode] = apply_filter_statespace(segment[selected_electrodes_names[electrode]], 
                A, B, C, D, Xnn[0,electrode])         
                for data_point in filt_result_temp:
                    filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]].append(data_point) 
            filters[f] = [A, B, C, D, Xnn]
    filtered_dataset = pd.DataFrame.from_dict(filter_results).transpose()    
    return filtered_dataset, filters

def unicorn_segmentation_overlap_withfilt(dataset, sample_duration, filters, selected_electrodes_names, freq_limits_names, 
    sampling_frequency, asr=None):#pipeline_type, 
                    
    window_hop = sampling_frequency//2
    segments = []
    labels = []
    outlier_labels = []
    dataset_c = copy.deepcopy(dataset)
    i = 0
    outliers = 0
    for frame_idx in range(sample_duration, dataset_c.shape[0], window_hop):
        if i == 0:
            temp_dataset = dataset_c.iloc[frame_idx-sample_duration:frame_idx, :-1] 
            # here apply filtering
            segment_filt, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        sample_duration, freq_limits_names, #pipeline_type, 
                                                            sampling_frequency, asr)
            
        else:
            # here only get 0.5 seconds extra, filter only that part, than concat with new part
            temp_dataset = dataset_c.iloc[frame_idx-window_hop:frame_idx, :-1] 
            # here apply filtering   
            segment_filt_new, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        window_hop, freq_limits_names, #pipeline_type, 
                                                                sampling_frequency, asr)
            if window_hop == sample_duration:
                segment_filt = segment_filt_new
            else:
                segment_filt = pd.concat([segment_filt.iloc[:,-(sample_duration-window_hop):].reset_index(drop=True), 
                segment_filt_new.reset_index(drop=True)], axis=1, ignore_index=True)

        if outlier > 0 or i == 0: 
            #when i ==0, filters are initiated so signal is destroyed. Dont use.
            #print(f'A segment was considered as an outlier due to bad signal in {outlier} channels')
            outliers +=1
            label_row = dataset_c.iloc[frame_idx-sample_duration:frame_idx, -1]
            label = label_row.value_counts()[:1]
            if (label[0] == sample_duration) and (label.index.tolist()[0] in ['0', '1']): #change here number of classes
                # 1 relax 2 right arm 3 left arm
                outlier_labels.append(int(label.index.tolist()[0]))
#                 current_seg = segment_filt.T
#                 current_seg.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
#                 #current_seg.drop(['CZ', 'C4', 'PZ', 'PO7', 'OZ'], axis=1, inplace=True)
#                 plt.rcParams['figure.figsize'] = (12, 8)
#                 plt.rcParams["legend.loc"] = 'upper right'
#                 current_seg.plot(subplots=True, legend=True, color='tab:blue', sharex=True, sharey=False)
#                 plt.xlabel('Datapoints of 2 seconds, captured with 250Hz', size=16)
#                 plt.ylabel('Processed EEG signal (\u03BCV)', size=16)
#                 plt.suptitle("Example of a 2 second segment of EEG signals",    # Set plot title
#                  size=25)
#                 plt.show()
        else:
            label_row = dataset_c.iloc[frame_idx-sample_duration:frame_idx, -1]
            label = label_row.value_counts()[:1]
            if (label[0] == sample_duration) and (label.index.tolist()[0] in ['0', '1']): 
                # 1 relax 2 right arm 3 left arm
                segments.append(segment_filt)
                labels.append(int(label.index.tolist()[0])) 
                #NOTE we need to have first class to be 0 for deepl pipeline
        i += 1
    label_amounts = Counter(labels)
    outlier_amounts = Counter(outlier_labels)
    print(f'amount of good segments: {len(labels)}')
    print(f"Good - relax: {label_amounts[0]}, legs: {label_amounts[1]}")
    print(f"Outliers - relax: {outlier_amounts[0]}, legs: {outlier_amounts[1]}")

#     # save output:
#     print(f"Good - relax: {label_amounts[0]}, legs: {label_amounts[3]}", 
#     file=open(f"{pipeline_type}_{freq_limits_names}_outliers.txt", "a"))
#     print(f"Outliers - relax: {outlier_amounts[0]}, legs: {outlier_amounts[3]}", 
#     file=open(f"{pipeline_type}_{freq_limits_names}_outliers.txt", "a"))

    return segments, labels

def plot_dataset(data_table, columns, match='like', display='line'):
    names = list(data_table.columns)
    # Create subplots if more columns are specified.
    if len(columns) > 1:
        f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
    else:
        f, xar = plt.subplots()
        xar = [xar]
    f.subplots_adjust(hspace=0.4)
    f.suptitle(r'EEG signals with Common Average Referencing', fontsize=16)
    # Pass through the columns specified.
    for i in range(0, len(columns)):
        xar[i].set_prop_cycle(color=['b', 'r', 'k', 'y', 'm'])
        # if a column match is specified as 'exact', select the column name(s) with an exact match.
        # If it's specified as 'like', select columns containing the name.

        # We can match exact (i.e. a columns name is an exact name of a columns or 'like' for
        # which we need to find columns names in the dataset that contain the name.
        if match[i] == 'exact':
            relevant_cols = [columns[i]]
        elif match[i] == 'like':
            relevant_cols = [name for name in names if columns[i] == name[0:len(columns[i])]]
        else:
            raise ValueError("Match should be 'exact' or 'like' for " + str(i) + ".")

        max_values = []
        min_values = []

        point_displays = ['+', 'x'] #'*', 'd', 'o', 's', '<', '>']
        line_displays = ['-'] #, '--', ':', '-.']
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


        # Pass through the relevant columns.
        for j in range(0, len(relevant_cols)):
        
            # Create a mask to ignore the NaN and Inf values when plotting:
            mask = data_table[relevant_cols[j]].replace([np.inf, -np.inf], np.nan).notnull()
            max_values.append(data_table[relevant_cols[j]][mask].max())
            min_values.append(data_table[relevant_cols[j]][mask].min())

            # Display point, or as a line
            if display[i] == 'points':
                xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                            point_displays[j%len(point_displays)])
                xar[i].set_xlabel('Datapoints (250Hz)')
                xar[i].set_ylabel('Label Value')
            else:
                xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                            line_displays[j%len(line_displays)])
                xar[i].set_xlabel('Datapoints (250Hz)')
                xar[i].set_ylabel('EEG signal (\u03BCV)')

        xar[i].tick_params(axis='y', labelsize=10)
        xar[i].legend(relevant_cols, fontsize='x-small', numpoints=1, loc='upper center',
                        bbox_to_anchor=(0.5, 1.25), ncol=len(relevant_cols), fancybox=True, shadow=True)

        xar[i].set_ylim([min(min_values) - 0.1*(max(max_values) - min(min_values)),
                            max(max_values) + 0.1*(max(max_values) - min(min_values))])

    # Make sure we get a nice figure with only a single x-axis and labels there.
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


    plt.show()