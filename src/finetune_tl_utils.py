import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import wandb
import pprint
import pickle
import os
import random

from meta_utils import subName, EarlyStopping
from pretrain_tl_utils import EEGNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_setup(config):
    print(f'Adding data for {config.test_subject}...')
    pre_path = "./preprocess/MIData_filtord2_freqlimits['1_100Hz']_ws500.pkl"
    a_file = open(pre_path, "rb")
    data_dict = pickle.load(a_file)
    X = data_dict['data']
    y = data_dict['labels']
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for df in X:
        for segment in range(len(X[df])): 
            # upperlimb classification
            if y[df][segment] == 0 or y[df][segment] == 3:
                if df in config.train_trials:
                    # put last trial of trial_num in validation set
                    X_train.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_train.append(y[df][segment]-2)
                    else :
                        y_train.append(y[df][segment])
                elif df == config.val_trials: 
                    #earlier trials in training
                    X_val.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_val.append(y[df][segment]-2)
                    else :
                        y_val.append(y[df][segment]) 
                elif df in config.test_trials:
                    X_test.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_test.append(y[df][segment]-2)
                    else :
                        y_test.append(y[df][segment]) 

    print(f'Length of X train: {len(X_train)}.')
    print(f'Length of X val: {len(X_val)}.')
    print(f'Length of X test: {len(X_test)}.')
    X_train_np = np.stack(X_train)
    X_val_np = np.stack(X_val)
    X_test_np = np.stack(X_test)
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)

    trainX = torch.from_numpy(X_train_np)
    trainY = torch.from_numpy(y_train_np)
    validationX = torch.from_numpy(X_val_np)
    validationY = torch.from_numpy(y_val_np)
    testX = torch.from_numpy(X_test_np)
    testY = torch.from_numpy(y_test_np)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)
    test = torch.utils.data.TensorDataset(testX, testY)

    trainloader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=config.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=config.batch_size, shuffle=True)

    return trainloader, valloader, testloader
    
def train_ft(config=None):
    # Initialize a new wandb run
    with wandb.init(project=f"DL_Finetune_X{config['test_subject']}_New", config=config):
        config = wandb.config
        pprint.pprint(config)
        trainloader, valloader, testloader = data_setup(config) # change data setup
        net = build_network(config)
        wandb.watch(net, log_freq=100)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
        early_stopping = EarlyStopping()
        # TRAINING
        for epoch in range(config.epochs):
            train_loss, train_acc, train_f1 = train_epoch(net, trainloader, optimizer)
            val_loss, val_acc, val_f1 = evaluate(net, valloader)
            wandb.log({"epoch": epoch,
            "train/train_loss": train_loss,
            "train/train_acc": train_acc,
            "train/train_f1": train_f1,
            "val/val_loss": val_loss,
            "val/vaL_acc": val_acc,
            "val/val_f1": val_f1})  

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        # TESTING
        test_loss, test_acc, test_f1 = evaluate(net, testloader)
        print(f'test loss: {test_loss}, test acc: {test_acc}, test f1: {test_f1}')
        wandb.summary['test_accuracy'] = test_acc
        wandb.summary['test_f1'] = test_f1
        
def build_network(config):
    net = EEGNET()
    sub = subName(config.test_subject)
    load_path = f'./models/DL_finetune/{sub}/{sub}_max_acc_legs_{config.val_subjects}_pt' # change this
    net.load_state_dict(torch.load(load_path))
    pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'trainable parameters: {pytorch_total_params_train}')
    return net.to(device)

def train_epoch(net, loader, optimizer):
    acc, running_loss, f1, batches, total = 0, 0, 0, 0, 0
    for _, (data, target) in enumerate(loader):
        data = data[:, np.newaxis, :, :]
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        # ➡ Forward pass
        loss = F.cross_entropy(net(data), target)
        running_loss += loss.item()
        _, predicted = torch.max(net(data).data, 1)
        acc += (predicted == target).sum().item()
        f1 += f1_score(target.data, predicted, average='macro')
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})
        batches += 1 
        total += target.size(0)
    running_loss = running_loss / len(loader)
    acc =  acc / total
    f1 =  f1 / batches
    return running_loss, acc, f1

def evaluate(net, loader):
    acc, running_loss, f1, batches, total = 0, 0, 0, 0, 0
    net.eval()
    for _, (data, target) in enumerate(loader):
        data = data[:, np.newaxis, :, :]
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        output = net(data)
        loss = F.cross_entropy(output, target)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        acc += (predicted == target).sum().item()
        f1 += f1_score(target.data, predicted, average='macro')
        batches += 1
        total += target.size(0)
    running_loss = running_loss / len(loader)
    acc =  acc / total
    f1 =  f1 / batches
    print(f"acc: {acc}")
    return running_loss, acc, f1