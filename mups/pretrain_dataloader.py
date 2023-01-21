''' Extracts Data from preprocessed pickle file based on subject requirement and loads them'''

from torch.utils.data import Dataset    
import pandas as pd
import numpy as np 
import pickle
import random

pd.options.mode.chained_assignment = None  # default='warn

def makeBigList(lst):
    biglst = []
    for item in lst :
        biglst = biglst + [k for k in range((item-1)*10,item*10)]
    return biglst

def makeRandomValList(val,config, mode = None):
    random.seed(val)
    trial_list= [ (val-1)*10 + k for k in range(0,10)]
    random.shuffle(trial_list)
    if mode == 'pre':
      return trial_list[:4],trial_list[4:6],trial_list[6:]
    elif mode == 'fine':
      return trial_list[:config['train_size']], trial_list[config['train_size']:config['train_size']+config['val_size']],trial_list[config['train_size']+config['val_size']:] 
    
# have make finetunepretraindata which takes in train subject list,val sub
# if pretrain return train data if label == train, val data if label == val

def data_gen(train, val, test, config, mode = None):
    
    random.seed(test)
    pre_path = "/content/drive/MyDrive/myMUPS/MIData_filtord2_freqlimits['1_100Hz']_ws500.pkl"
    a_file = open(pre_path, "rb")
    data_dict = pickle.load(a_file)
    X = data_dict['data']
    y = data_dict['labels']
    
    val_list = makeRandomValList(val, config, mode = 'pre')
    #print(val_list)

    data = []
    label = []
    if mode == 'train':
        for df in X:
            if df in makeBigList(train)+val_list[0]:
                for segment in range(len(X[df])):
                    # only legs and relax
                    if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                        data.append(X[df][segment])
                        if y[df][segment] == 3:
                            label.append(y[df][segment]-2)
                        else :
                            label.append(y[df][segment])
                            
    elif mode == 'val':
        for df in X:
            if df in val_list[1]:
              for segment in range(len(X[df])):
                  # only legs and relax
                  if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                      data.append(X[df][segment])
                      if y[df][segment] == 3:
                          label.append(y[df][segment]-2)
                      else :
                          label.append(y[df][segment])
                    
    elif mode == 'test':
        for df in X:
            if df in val_list[2]:
                for segment in range(len(X[df])):
                    # only legs and relax
                    if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                        data.append(X[df][segment])
                        if y[df][segment] == 3:
                            label.append(y[df][segment]-2)
                        else :
                            label.append(y[df][segment])                    
    
    
    data = np.stack(data)
    label = np.stack(label)
    
    return data, label

class LoadmyMUPSptdata(Dataset):

    def __init__(self, setname, train, val, test, config, train_aug=False):
        self.config = config

        if setname == 'train':
            print('preparing training data')
            # generating training data
            train_dataset = data_gen(train, val, test, config, mode = setname)
            train_X = train_dataset[0] # data
            train_y = train_dataset[1] # label

            # shuffling training data
            idx = list(range(len(train_y)))
            np.random.shuffle(idx)
            train_X = train_X[idx]
            train_y = train_y[idx]

            # doing load MUPS data operations
            train_y = train_y.ravel()
            train_win_x = train_X[:, np.newaxis, :, :].astype('float32')
            train_win_y=train_y

            print('Train data :',train_win_x.shape)
            print('Train labels :',train_win_y.shape)           
            # outputting data for operations
            self.data=train_win_x
            self.label=train_win_y

        elif setname == 'val':
            print('Preparing validation data')
            # generating validation data
            val_dataset = data_gen(train, val, test, config, mode = setname)
            val_X = val_dataset[0] # data
            val_y = val_dataset[1] # label

            # shuffling training data
            idx = list(range(len(val_y)))
            np.random.shuffle(idx)
            val_X = val_X[idx]
            val_y = val_y[idx]

            # doing load MUPS data operations
            val_y = val_y.ravel()
            val_win_x = val_X[:, np.newaxis, :, :].astype('float32')
            val_win_y= val_y

            # outputting data for operations
            self.data = val_win_x
            self.label=val_win_y
            print('Val data :',val_win_x.shape)
            print('Val labels :',val_win_y.shape)
            self.X_val=val_win_x
            self.y_val=val_win_y

        elif setname == 'test':
            print('Preparing test data')
            # generating test data
            test_dataset = data_gen(train, val, test, config, mode = setname)
            test_X = test_dataset[0] # data
            test_y = test_dataset[1] # label

            # shuffling training data
            idx = list(range(len(test_y)))
            np.random.shuffle(idx)
            test_X = test_X[idx]
            test_y = test_y[idx]

            # doing load MUPS data operations
            test_y = test_y.ravel()
            test_win_x = test_X[:, np.newaxis, :, :].astype('float32')
            test_win_y= test_y
            
            print('Test data :',test_win_x.shape)
            print('Test labels :',test_win_y.shape)
            # outputting data for operations
            self.data=test_win_x
            self.label=test_win_y

        self.num_class=self.config['way']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label=self.data[i], self.label[i]
        return data, label