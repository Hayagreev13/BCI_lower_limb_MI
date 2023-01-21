""" Trainer for pretrain phase. """

import tqdm
import numpy as np
from sklearn.metrics import  precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import src.meta_utils as utils
from pretrain_dataloader import LoadmyMUPSptdata
from samplers import CategoriesSampler
from meta_learner import MtlLearner
from modelsaver import save_model

  
def pre_train(config=None):
    """The function for the pre-train phase."""

    # Set the pretrain log
    trlog = {}
    #trlog['config'] = vars(self.config)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    # Set the timer
    timer = utils.Timer()
    # Set global count to zero
    global_count = 0

    # early stopping to avoid overfitting of data - 7 epochs in question
    early_stopping = utils.EarlyStopping()
#    LoadmyMUPSptdata --> def __init__(self, setname, train, val, test, config, train_aug=False):
    # Load pretrain set
    print("Preparing dataset loader")
    trainset = LoadmyMUPSptdata('train', config['train'], config['val'][0], config['test'][0], config, train_aug=False)
    train_loader = DataLoader(dataset=trainset, batch_size=config['pre_batch_size'], shuffle=True, num_workers=2, pin_memory=True)

    # Load meta-val set
    valset = LoadmyMUPSptdata('val', config['train'], config['val'][0], config['test'][0], config)
    val_sampler = CategoriesSampler(valset.label, 20, config['way'], config['shot'] + config['val_query'])
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=2, pin_memory=True)

    # Set pretrain class number 
    #num_class_pretrain = trainset.num_class
    
    # Build pretrain model
    model = MtlLearner(config, mode='pre')
    #model=model.float()
    # Set optimizer
    params=list(model.encoder.parameters())+list(model.pre_fc.parameters()) 
    optimizer=optim.Adam(params)
    
    # Set model to GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    # wandb.watch(model, log_freq=100)
    # Start pretrain
    for epoch in range(1, config['pre_max_epoch'] + 1): #30
        # Set the model to train mode

        print('Epoch {}'.format(epoch))
        model.train()
        model.mode = 'pre'
        # Set averager classes to record training losses and accuracies
        train_loss_averager = utils.Averager()
        train_acc_averager = utils.Averager()
            
        # Using tqdm to read samples from train loader
        

        tqdm_gen = tqdm.tqdm(train_loader)
        #for i, batch in enumerate(train_loader):
        for i, batch in enumerate(tqdm_gen, 1):
            # Update global count number 
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            label = batch[1]
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            logits = model(data) # modified
            loss = F.cross_entropy(logits, label)
            # Calculate train accuracy
            acc = utils.count_acc(logits, label)
            # Print loss and accuracy for this step
            train_loss_averager.add(loss.item())
            train_acc_averager.add(acc)
            # Loss backwards and optimizer updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the averagers
        train_loss_averager = train_loss_averager.item()
        train_acc_averager = train_acc_averager.item()
        
        # start the original evaluation
        model.eval()
        model.mode='origval'
        #print('valset.y_val',valset.y_val)
        _, valid_results=val_orig(model,valset.X_val,valset.y_val)
        print ('validation accuracy ', valid_results[0])
        
        # Start validation for this epoch, set model to eval mode
        model.eval()
        model.mode = 'preval'

        # Set averager classes to record validation losses and accuracies
        val_loss_averager = utils.Averager()
        val_acc_averager = utils.Averager()

        # Generate the labels for test 
        label = torch.arange(config['way']).repeat(config['val_query'])
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(config['way']).repeat(config['shot'])
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
          
        # Run meta-validation
        for i, batch in enumerate(val_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            #data=data.float()
            p = config['shot'] * config['way']
            data_shot, data_query = data[:p], data[p:]
            logits = model((data_shot, label_shot, data_query)) # modified
            loss = F.cross_entropy(logits, label)
            acc = utils.count_acc(logits, label)
            val_loss_averager.add(loss.item())
            val_acc_averager.add(acc)

        # Update validation averagers
        val_loss_averager = val_loss_averager.item()
        val_acc_averager = val_acc_averager.item()       

        # Update best saved model
        if val_acc_averager > trlog['max_acc']:
            trlog['max_acc'] = val_acc_averager
            trlog['max_acc_epoch'] = epoch
            save_model(model,config['test'][0],'pretrain',config['chosen_val'],mode='pt')

        # Update the logs
        trlog['train_loss'].append(train_loss_averager)
        trlog['train_acc'].append(train_acc_averager)
        trlog['val_loss'].append(val_loss_averager)
        trlog['val_acc'].append(val_acc_averager)

        if epoch % 5 == 0: # change to 5
            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / config['pre_max_epoch'])))

        early_stopping(val_loss_averager)
        if early_stopping.early_stop:
            break

    return epoch, train_loss_averager, train_acc_averager, val_loss_averager, val_acc_averager, trlog['max_acc']

def val_orig(model,X_val, y_val):
    predicted_loss=[]
    inputs = torch.from_numpy(X_val)
    labels = torch.FloatTensor(y_val*1.0)
    inputs, labels = Variable(inputs), Variable(labels)
    
    results = []
    predicted = []
            
    model.eval()
    model.mode = 'origval'
    

    if torch.cuda.is_available():
        inputs= inputs.type(torch.cuda.FloatTensor)
    else:
        inputs = inputs.type(torch.FloatTensor)

    predicted=model(inputs)
    predicted= predicted.data.cpu().numpy()

    Y=labels.data.numpy()
    predicted=np.argmax(predicted, axis=1)        
    for param in ["acc", "auc", "recall", "precision","fmeasure"]:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted), average='micro'))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted), average='micro')
            recall = recall_score(Y, np.round(predicted), average='micro')
            results.append(2*precision*recall/ (precision+recall))
    
    return predicted, results