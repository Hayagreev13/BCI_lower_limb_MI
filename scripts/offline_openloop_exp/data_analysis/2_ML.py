import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt

import src.offline_utils as utils


pd.options.mode.chained_assignment = None  # default='warn'

def main():
    for subj in args.subjects:
        print(subj)
        if 'csp' in args.pline:
            # filterbank
            list_of_freq_lim = [[[5, 10], [10, 15], [15, 20], [20, 25]]]
            freq_limits_names_list = [['5_10Hz','10_15Hz','15_20Hz', '20_25Hz']]
            filt_orders = [2]
            window_sizes = [500]
            execute_ml('csp', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj)
        if 'deep' in args.pline:
            list_of_freq_lim = [[[1,100]]]
            freq_limits_names_list = [['1_100Hz']]
            filt_orders = [2]
            window_sizes = [500]
            execute_ml('deep', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj)
        if 'riemann' in args.pline:
            list_of_freq_lim = [[[8, 35]]]
            freq_limits_names_list = [['8_35Hz']]
            filt_orders = [2]
            window_sizes = [500]
            execute_ml('riemann', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj)

def execute_ml(pipeline_type, list_of_freq_lim, freq_limits_names_list, filt_ord, window_size, subject):
    print(f'Initializing for {pipeline_type} experimentation...')
    # INIT
    sampling_frequency = 250 
    # testing here for 8 electrodes:
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    #file_elec_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
    n_electrodes = len(electrode_names)

    #subject = 'X01'
    method = 'wet'
    #folder_path = Path(f'./data/Subjects/{subject}/openloop')
    folder_path = Path(f'./data_collection/Expdata/Subjects/{method}/{subject}/openloop')
    print(folder_path)
    #env_noise_path = Path(f'./data/Subjects/{subject}/Envdata')
    result_path = Path(f'./Results/ML/{subject}_{method}')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'{subject}_relaxlegs_{pipeline_type}_filt_exp.csv'
    num_classes = 2
    
    dataset_full = {}
    trials_amount = 0
    asr = None

    for instance in os.scandir(folder_path):
        if instance.path.endswith('.csv'): 
            trials_amount +=1
            print(f'adding_{instance} to dataset...')
            sig = pd.read_csv(instance.path)
            X = sig.loc[:,electrode_names]
            y = sig.loc[:,'Class']
            dataset_full[str(instance)] = pd.concat([X,y], axis=1)
            #print(pd.value_counts(y))
    results = {}   
    for freq_limit_instance in range(len(list_of_freq_lim)):
        freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
        freq_limits_names = freq_limits_names_list[freq_limit_instance]
        filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
        X_train, y_train, X_val, y_val  = [], [], [], []
        X_all, y_all = [], []
        df_num = 0
        print(f'experimenting with filter order of {filt_ord}, freq limits of {freq_limits_names}, \
                and ws of {window_size}.')
        '''
        for df in dataset_full:
            X_segmented, y = segmentation_all(dataset_full[df],sample_duration)
            for segment in range(len(X_segmented)):
                segment_filt = filter_1seg(X_segmented[segment].transpose(), selected_electrodes_names,filters, sample_duration,
                    freq_limits_names)

                segment_filt = segment_filt.transpose()
                # append each segment-df to a list of dfs
                X_all.append(segment_filt)
                y_all.append(y[segment])
                print(f'Current length of X: {len(X_all)}.')

            # transform to np for use in ML-pipeline
            X_np = np.stack(X_all)
            y_np = np.array(y_all).ravel()
        '''
        for df in dataset_full:
            X_temp, y_temp = utils.unicorn_segmentation_overlap_withfilt(dataset_full[df], window_size, filters,
            electrode_names, freq_limits_names, pipeline_type, sampling_frequency, asr)
            for segment in range(len(X_temp)):
                X_all.append(X_temp[segment])
                y_all.append(y_temp[segment])
                if df_num == 3 or df_num == 7: # > 0 and df_num % 5 == 0: 
                    X_val.append(X_temp[segment])
                    y_val.append(y_temp[segment]) 
                else:
                    X_train.append(X_temp[segment])
                    y_train.append(y_temp[segment]) 
            df_num += 1
            print(f'Current length of X train: {len(X_train)}.')
            print(f'Current length of X val: {len(X_val)}.')

        X_train_np = np.stack(X_train)
        X_val_np = np.stack(X_val)
        y_train_np = np.array(y_train)
        y_val_np = np.array(y_val)
        print(f"shape training set: {X_train_np.shape}")
        print(f"shape validation set: {X_val_np.shape}")
        
        X_np = np.stack(X_all)
        y_np = np.array(y_all).ravel()
        print(f"Cross Validation set length: {X_np.shape}")


        # gridsearch experimentation
        chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
        for clf in chosen_pipelines:
            print(f'applying {clf} with gridsearch...')
            acc, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(X_train_np, y_train_np, 
            X_val_np, y_val_np, chosen_pipelines, clf)
            results[f"grid_search_{clf}_{filt_ord}_{freq_limits_names}_{window_size}"] = {
            'clf': clf, 'filter_order' : filt_ord, 'freq_limits' : freq_limits_names, 
            'windowsize' : window_size, 'test_accuracy': acc, 'acc_classes': acc_classes, 
            'test_f1' : f1, 'time (seconds)': elapsed_time}

        # CV experimentation
        chosen_pipelines = utils.init_pipelines(pipeline_type)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'f1': 'f1', 'acc': 'accuracy','prec_macro': 'precision_macro','rec_macro': 'recall_macro'}
        for clf in chosen_pipelines:
            print(f'applying {clf}...')
            start_time = time.time()
            scores = cross_validate(chosen_pipelines[clf], X_np, y_np, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
            elapsed_time = time.time() - start_time
            results[f"cv_{clf}_{filt_ord}_{freq_limits_names}_{window_size}"] = {
            'clf': clf, 'filter_order' : filt_ord, 'freq_limits' : freq_limits_names, 
            'windowsize' : window_size, 'test_accuracy': scores['test_acc'].mean(),'test_f1': scores['test_f1'].mean(),
            'test_prec': scores['test_prec_macro'].mean(), 'test_rec': scores['test_rec_macro'].mean(), 
            'time (seconds)': elapsed_time, 'train_accuracy': scores['train_acc'].mean() } 

    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_accuracy', ascending=False) 
    sub_col = [subject]*results_df.shape[0]
    results_df.insert(loc=0, column='subject', value=sub_col)
    results_df.to_csv(result_path / results_fname)
    print('Finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used for after preprocessing. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    parser.add_argument("--subjects", nargs='+', default=['X01'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    args, unparsed = parser.parse_known_args()
    main()