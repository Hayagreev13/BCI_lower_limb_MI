'''
Code for loading data from csv file and preprocessing it with filters, cleaning and 
saving the data of 9 subjects in one pickle file
End product after preprocessing is a pickle containing data_dict with data and labels 
End shape of data is ~ 90x104x8x500
End shape of labels is ~ 90x104 
End product after data_gen is shuffled train and test data saved as a .mat file
'''

#from meegkit.asr import ASR
#import time

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

import src.unicorn_utils as utils

pd.options.mode.chained_assignment = None  # default='warn'

def data_preprocessing(list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes):
    print(f'Preprocessing for meta learning experimentation...')
    # INIT
    sampling_frequency = 250 
    # testing here for 8 electrodes:
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    asr = False
    #folder_path = Path(f'./data/Subjects/{subject}/openloop')
    #env_noise_path = Path(f'./data/Subjects/{subject}/Envdata')
    result_path = Path(f'preprocess/test/')
    result_path.mkdir(exist_ok=True, parents=True)  
    dataset_full = {}
    trials_amount = 0
    asr = None
    method = 'wet'
    #subjects = ['X01','X02','X03','X04','X05','X06','X07','X08','X09']
    subjects = ['X01'] #dry subjects
    for subject in subjects :
        #folder_path = Path(f'./data/Subjects/{subject}/openloop')
        folder_path = Path(f'Expdata/Subjects/{method}/{subject}/openloop')
        for instance in os.scandir(folder_path):
            if instance.path.endswith('.csv'): 
                trials_amount +=1
                #print(f'adding_{instance} to dataset...')
                sig = pd.read_csv(instance.path)
                X = sig.loc[:,electrode_names]
                y = sig.loc[:,'Class']
                dataset_full[str(instance)] = pd.concat([X,y], axis=1)
        print(f'{subject} data loaded to dataset & total length - {len(dataset_full)}')
  
    for window_size in window_sizes:
        for filt_ord in filt_orders:
            for freq_limit_instance in range(len(list_of_freq_lim)):
                freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
                freq_limits_names = freq_limits_names_list[freq_limit_instance]
                print(f'experimenting with filter order of {filt_ord}, freq limits of {freq_limits_names},and ws of {window_size}.')

                filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
                data_dict = {'data' : {}, 'labels': {}}

                df_num = 0
                for df in dataset_full:
                    print(df)
                    data_dict['data'][df_num], data_dict['labels'][df_num] = [], []
                    X_temp, y_temp = utils.unicorn_segmentation_overlap_withfilt(dataset_full[df], window_size, filters,
                    electrode_names, freq_limits_names, #pipeline_type, 
                                                                           sampling_frequency, asr)
                    for segment in range(len(X_temp)): 
                        data_dict['data'][df_num].append(X_temp[segment])
                        data_dict['labels'][df_num].append(y_temp[segment]) 
                    df_num += 1

                print(f'Saving file with filter order of {filt_ord}, freq limits of {freq_limits_names},and ws of {window_size}.')
                results_fname = f'{subject}_{method}_filtord{filt_ord}_freqlimits{freq_limits_names}_ws{window_size}.pkl'
                save_file = open(result_path / results_fname, "wb")
                pickle.dump(data_dict, save_file)
                save_file.close()
                print('Finished a preprocess pipeline.')

def data_gen(subject_list, string, clstype, freq_limit):

    pre_path = Path(f'./data/preprocess/')
    for instance in os.scandir(pre_path):
        if freq_limit in str(instance) :
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
        
    data = []
    label = []
    for df in X:
        if df in utils.makeBigList(subject_list):
            for segment in range(len(X[df])):
                if clstype == 'multiclass':
                    data.append(X[df][segment])
                    label.append(y[df][segment])
                elif clstype == 'legs':
                    # only legs and relax
                    if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                        data.append(X[df][segment])
                        if y[df][segment] == 3:
                            label.append(y[df][segment]-2)
                        else :
                            label.append(y[df][segment])    
    
    if clstype == 'multiclass':
        print('Generating multiclass data')
    elif clstype == 'legs':
        print('Generating data for relax vs legs')
        
    print(f'Final length of {string} data :', len(data))
    print(f'Final length of {string} labels :',len(label))
    
    data = np.stack(data)
    label = np.stack(label)
    
    return data, label

if __name__ == '__main__':
    # preprocessing all the csv files and saving all of them as a pickle

    parser = argparse.ArgumentParser(description="Select Validation subjects for model training")
    parser.add_argument("--val", nargs='+', default=[4,6], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    parser.add_argument("--clstype", type=str, default='legs', choices=['multiclass', 'legs'])  # Classification type
    parser.add_argument("--freq", type=str, default='100', choices=['100','35'])  # select which freq prep file to use
    args = parser.parse_args()

    list_of_freq_lim = [[[1,100]]]#, [[8,35]]]
    freq_limits_names_list = [['1_100Hz']]#, '8_35Hz']
    filt_orders = [2]
    window_sizes = [500]    
    data_preprocessing(list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)