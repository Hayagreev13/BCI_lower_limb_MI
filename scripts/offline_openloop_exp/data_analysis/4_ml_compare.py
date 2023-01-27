import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import pickle
import scipy.io
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt
import random

import src.offline_utils as utils

def execution(pipeline_type, subject, classes):
    print(f'Initializing for {pipeline_type} machine learning...')
    method = 'wet'
    folder_path = Path(f'./data_collection/Expdata/Subjects/{method}/{subject}/openloop')
    result_path = Path(f'./Results/Compare/{subject}_{classes}classes')
    result_path.mkdir(exist_ok=True, parents=True)

    results = {}
    #loading data
    for instance in os.scandir(folder_path):
        if pipeline_type in instance.path and subject in instance.path: 
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
            train_acc_cv, val_acc_cv, val_prec_cv, val_rec_cv, train_f1_cv, val_f1_cv, \
            val_roc_auc_cv, acc_classes_cv = {}, {}, {}, {}, {}, {}, {}, {}
            break

    trials = [0,1,2,3,4,5,6,7,8,9]
    random.seed(int(subject[-1]))
    for trial_num in range(1,5):
        total = 5
        all_trial_list = []
        while len(all_trial_list) < total:
            trial_list = random.sample(trials, len(trials)) 
            if trial_list not in all_trial_list:    
                all_trial_list.append(trial_list)
                train_trials = trial_list[:trial_num]
                val_trials = trial_list[trial_num] #not used for ML
                test_trials = trial_list[5:]
                print(f"{train_trials}, {val_trials}, {test_trials}")
                X_train, y_train, X_test, y_test = [], [], [], []
                for df in X:
                    for segment in range(len(X[df])): 
                        # upperlimb classification
                        if classes ==4:
                            if y[df][segment] == 0 or y[df][segment] == 1 or y[df][segment] == 2 or y[df][segment] == 3:
                                if df in train_trials:
                                    #earlier trials in training
                                    X_train.append(X[df][segment])
                                    y_train.append(y[df][segment])  
                                elif df in test_trials:
                                    X_test.append(X[df][segment])
                                    y_test.append(y[df][segment])
                        elif classes == 2:
                            if y[df][segment] == 0  or y[df][segment] == 3:
                                if df in train_trials:
                                    #earlier trials in training
                                    X_train.append(X[df][segment])
                                    y_train.append(y[df][segment])  
                                elif df in test_trials:
                                    X_test.append(X[df][segment])
                                    y_test.append(y[df][segment])                            
 
                print(f'Current length of X train: {len(X_train)}.')
                print(f'Current length of X test: {len(X_test)}.')
                X_train_np = np.stack(X_train)
                X_test_np = np.stack(X_test)
                y_train_np = np.array(y_train)
                y_test_np = np.array(y_test)

                # gridsearch experimentation csp or riemann
                chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
                for clf in chosen_pipelines:
                    print(f'applying {clf} with gridsearch...')
                    test_accuracy, prec, rec, roc_auc, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(
                        X_train_np, y_train_np, X_test_np, y_test_np, chosen_pipelines, clf)
                    if clf not in val_acc_cv:
                        val_acc_cv[clf],val_prec_cv[clf],val_rec_cv[clf],val_f1_cv[clf],val_roc_auc_cv[clf],\
                            acc_classes_cv[clf] = [],[],[],[],[],[]
                    val_acc_cv[clf].append(test_accuracy)
                    val_prec_cv[clf].append(prec)
                    val_rec_cv[clf].append(rec)
                    val_f1_cv[clf].append(f1)
                    val_roc_auc_cv[clf].append(np.array(roc_auc).mean()) 
                    acc_classes_cv[clf].append(acc_classes)
                
                    results[f"run_{len(all_trial_list)}_{clf}_crossval_{instance.path}"] = {'final_test_accuracy': np.around(np.array(val_acc_cv[clf]).mean(),3), 
                        'final_test_f1': np.around(np.array(val_f1_cv[clf]).mean(),3), 'test_prec': np.around(np.array(val_prec_cv[clf]).mean(),3), 
                    'test_rec': np.around(np.array(val_rec_cv[clf]).mean(),3), 'test_roc_auc': np.around(np.array(val_roc_auc_cv[clf]).mean(),3), 
                    'full_testacc': val_acc_cv[clf], 'full_acc_classes': acc_classes_cv[clf]}  
                print('Finished 1 pipeline')

        results_fname = f'{pipeline_type}_{subject}_trialnum_{trial_num}.csv'
        results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('final_test_accuracy', ascending=False)  
        results_df.to_csv(result_path / results_fname)
    print('Finished')

subjects = ['X0'+str(i) for i in range(1,10)]
pipelines = ['csp','riemann']
classes = [2,4]
for subj in subjects:
    print(subj)
    for pline in pipelines:
        print(pline)
        for cls in classes:
            execution(pline, subj, cls)