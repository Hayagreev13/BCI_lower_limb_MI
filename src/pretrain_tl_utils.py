import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import wandb
import pprint
import random
import copy
import os
import os.path as osp
import numpy as np
import pickle

from meta_utils import subName, makeBigList

# Set the GPU id
if torch.cuda.is_available(): 
  print('Using device :', torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field, filter_sizing, mean_pool, activation_type, dropout, D):
        super(EEGNET,self).__init__()
        sample_duration = 500
        channel_amount = 8
        num_classes = 3
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 320
        self.fc2 = nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        prediction = self.fc2(out)
        return prediction

def data_setup_pt(batch_size, val, test):
    
    train = [k for k in range(1,10)]
    train.remove(val)
    train.remove(test)
    pre_path = "./preprocess/MIData_filtord2_freqlimits['1_100Hz']_ws500.pkl"
    a_file = open(pre_path, "rb")
    data_dict = pickle.load(a_file)
    X = data_dict['data']
    y = data_dict['labels']
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
      
    for df in X:
        if df in makeBigList(train):
            for segment in range(len(X[df])):
                # only legs and relax
                if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                    X_train.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_train.append(y[df][segment]-2)
                    else :
                        y_train.append(y[df][segment])

        elif df in makeBigList([val])[:4]:       
            for segment in range(len(X[df])):
                # only legs and relax
                if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                    X_val.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_val.append(y[df][segment]-2)
                    else :
                        y_val.append(y[df][segment])

        elif df in makeBigList([val])[4:]:       
            for segment in range(len(X[df])):
                # only legs and relax
                if y[df][segment] == 0 or y[df][segment] == 3: # label for relax is 0, and legs is 3
                    X_test.append(X[df][segment])
                    if y[df][segment] == 3:
                        y_test.append(y[df][segment]-2)
                    else :
                        y_test.append(y[df][segment])                        
                  
    X_train_np = np.stack(X_train)
    X_val_np = np.stack(X_val)
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    X_test_np = np.stack(X_test)
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
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader

def train_pt(config=None):
    # Initialize a new wandb run
    with wandb.init(project=f"DL_PreTrain_X{config['test_subject']}_New",config=config):
        config = wandb.config
        pprint.pprint(config)
        trainloader, valloader, testloader = data_setup_pt(config.batch_size, config.val_subjects, config.test_subject)
        net = build_network_pt(config)
        wandb.watch(net, log_freq=100)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
        for epoch in range(config.epochs):
            print(f"Epoch num: {epoch}")
            train_loss, train_acc, train_f1 = train_epoch_pt(net, trainloader, optimizer) # add f1
            #first_params = net.parameters
            val_loss, val_acc, val_f1, ft_acc = evaluate_pt(net, valloader, testloader, optimizer) # add f1
            #print(f"This should be True: {net.parameters == first_params}")
            wandb.log({"epoch": epoch,
            "train/train_loss": train_loss,
            "train/train_acc": train_acc,
            "train/train_f1": train_f1,
            "val/val_loss": val_loss,
            "val/vaL_acc": val_acc,
            "val/val_f1": val_f1,
            "val_finetune_acc": ft_acc})

        save_model(net,config.test_subject,config.val_subjects, mode = 'pt')

def build_network_pt(config):
    net = EEGNET()
    pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'trainable parameters: {pytorch_total_params_train}')
    return net.to(device)

def train_epoch_pt(net, loader, optimizer):
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
        f1 += f1_score(target.cpu().data, predicted, average='macro')
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

def evaluate_pt(net, valloader, testloader, optimizer):
    acc, running_loss, f1, batches, total = 0, 0, 0, 0, 0
    current_net = copy.deepcopy(net)
    current_opt = copy.deepcopy(optimizer)
    current_net.eval()
    for epoch in range(5):
        valacc, valrunning_loss, valf1, valbatches, valtotal = 0, 0, 0, 0, 0
        for _, (data, target) in enumerate(valloader):
            data = data[:, np.newaxis, :, :]
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = F.cross_entropy(current_net(data), target)
            valrunning_loss += loss.item()
            _, predicted = torch.max(current_net(data).data, 1)
            valacc += (predicted == target).sum().item()
            valf1 += f1_score(target.data, predicted, average='macro')
            loss.backward()
            current_opt.step()
            valbatches += 1
            valtotal += target.size(0)
        print(f'valacc epoch {epoch}: {valacc / valtotal}')
    #print(f"This should be false: {current_net.parameters() == net.parameters()}")
    #print(f"This should be false: {current_opt == optimizer}")

    for _, (data, target) in enumerate(testloader):
        data = data[:, np.newaxis, :, :]
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        output = current_net(data)
        loss = F.cross_entropy(output, target)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        acc += (predicted == target).sum().item()
        f1 += f1_score(target.data, predicted, average='macro')
        batches += 1
        total += target.size(0)

    running_loss = running_loss / len(testloader)
    acc =  acc / total
    f1 =  f1 / batches
    print(f"test acc: {acc}, test f1: {f1}")
    ft_acc = valacc / valtotal

    return running_loss, acc, f1, ft_acc

def save_model(model,sub,val,mode):
        """The function to save checkpoints.
        Args:
          model : ML model
          sub : subject
          val : validation set values
          mode : either pt for pretraining and ft for finetuning
        """  
        sub = subName(sub)
        save_path = f'./models/DL_finetune/'
        if not osp.exists(save_path):
            os.mkdir(save_path)
        save_path = osp.join(save_path, sub)
        if not osp.exists(save_path):
            os.mkdir(save_path)
         #torch.save(net.state_dict(), f'pretrain_models/{config.test_subject}/EEGNET_ft_v2/EEGNET-PreTrain_val{config.val_subjects[0]}')
        torch.save(model.state_dict(), osp.join(save_path, f'{sub}_max_acc_legs_{val}_{mode}'))