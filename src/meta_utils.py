""" Additional utility functions & Tools for GPU. """
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def set_gpu(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', cuda_device)

def Sort_Tuple(tup):
 
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key = lambda x: x[1], reverse = True)
    return tup

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test) 
    y_pred = lb.transform(y_pred) 
    return roc_auc_score(y_test, y_pred, average=average)

def GetBestDict(lst):
  max_acc = 0
  max_dict = {}
  for pair in lst:
    print('Pre Test Accuracy:',pair[0][1])
    print('Actual Test Accuracy:',pair[1])
    if pair[1] > max_acc :
      max_acc = pair[1]
      max_dict = pair[0][0]
  print('Returning max accuracy dict', max_acc)
  return max_dict, max_acc

def subName(test):
    if test < 10:
        sub = 'X0'+str(test)
    else :
        sub = 'X'+str(test)
    return sub
    
def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=1e-4):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True