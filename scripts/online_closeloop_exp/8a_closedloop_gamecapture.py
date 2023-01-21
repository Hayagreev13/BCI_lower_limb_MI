#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lab streaming layer library to capture the data sent by unicorn EEG headset
from pylsl import StreamInlet, resolve_stream
from pynput.keyboard import Key, Controller

#numpy and pd for data storing and manipulation
import numpy as np
import pandas as pd

# misc libraries to structure the cues properly and save it with date time and stuff
from datetime import date
from pathlib import Path
import os

import torch
import src.realtime_utils as utils

keyboard_game = Controller()

def run_game(subject):
    expName = 'game'
    exType = 'wet'
    expInfo = {'participant': subject,'type': exType, 'expName' : expName}

    #INIT
    filt_ord = 2
    freq_limits = np.asarray([[1,100]]) 
    freq_limits_names = ['1_100Hz']
    sample_duration = 125
    sampling_frequency = 250
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
    segments, labels, predictions = [], [], []
    total_outlier = 0

    # init DL model
    net = utils.FeatureExtractor()
    path = Path(f'./models/{subject}_metamodel.pth')
    pretrained_dict = torch.load(path)
    net.load_state_dict(pretrained_dict)
    net = net.float()
    net.eval()

    #change path of folders according to your needs
    # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc

    result_path = Path(f'./Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expName+'/')
    result_path.mkdir(exist_ok=True, parents=True)

    columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
                                    'Battery','Counter','Validation']

    data_dict = dict((k, []) for k in columns)
    current_seg = pd.DataFrame()
    total_MI_outliers = 0
    all_MI_segments, all_MI_labels, predictions = [], [], []

    MI_dict = {'MI_segments' : [], 'predictions': []}

    # below code is for initializing the streaming layer which will help us capture data later
    finished = False
    streams = resolve_stream()
    inlet = StreamInlet(streams[0])

    # Auto updating trial numbers
    trial_list = []
    for instance in os.scandir(result_path):
            if instance.path.endswith('.csv'):
                length = len(instance.path)
                trial_list.append(int(instance.path[length-5]))

    if len(trial_list) == 0:
        session = '01'
    elif len(trial_list) < 9 :
        session = len(trial_list) + 1
        session = '0' + str(session)
    else :
        session = str(len(trial_list) + 1)

    print(f"Conducting {expName} experiment for subject :", expInfo['participant'])
    print('No. of Practice Trials before :', 2)
    print("Trial Number :", session)

    print('Actual Trial')
    print('Total number of trials as of now :', int(session) + 2)
    results_fname = expInfo['participant']+'_'+str(date.today())+'_'+expName+'_'+ expInfo['type']+'_'+session+'.csv'
    print("Saving file as .. ", results_fname)


    Fs = 250 # sampling frequency of Unicorn EEG cap
    initial = 0
    final = 125
    prediction = -1
    outliers = []
    key = False
    while not finished:

        sample, timestamp = inlet.pull_sample()
        
        res = [timestamp] + sample 
        data_dict = utils.update_data(data_dict,res)

        if len(data_dict['FZ']) % 125 == 0:
            df, initial, final = utils.segment_dict(initial, final, sample_duration, data_dict)
            segment_filt, out, filters = utils.pre_processing(df, electrode_names, filters, 
                            sample_duration, freq_limits_names, sampling_frequency)
            current_seg = utils.concatdata(current_seg,segment_filt)   
            outliers.append(out)   
            # do prediction with current segment and update number
            if current_seg.shape[1] == 500:
                if sum(outliers) > 0:
                    total_MI_outliers +=1
                    print('OUTLIER')
                    if key:
                        keyboard_game.release(key) 
                else:
                    all_MI_segments.append(current_seg)
                    prediction = utils.do_prediction(current_seg, net)
                    predictions.append(int(prediction[0]))
                    print(f"prediction: {prediction[0]}") 
                    if key:
                        keyboard_game.release(key)

                    if prediction[0] ==0:
                        key = Key.up
                        keyboard_game.press(key)
                    elif prediction[0] ==1:
                        key = Key.down
                        keyboard_game.press(key)

                outliers = outliers[1:]