""" Trainer for meta-train phase. """
import os.path as osp
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import time
import wandb

import src.meta_utils as utils
from pretrain_dataloader import LoadmyMUPSftdata
from samplers import CategoriesSampler
from meta_learner import MtlLearner
from modelsaver import save_model
from pretrainer import pre_train
from feature_extractor import MindReader


def meta_finetune(config=None, mode='finetune'):
    """The function for the meta-train phase."""
    # initializing wandb for recording
    with wandb.init(project=f"Realtime testing",config=config): #change project name as required
      config = wandb.config

      if mode == 'pretrain':
        # getting pretraining results and saving model
        pre_epoch, pre_train_loss, pre_train_acc, pre_val_loss, pre_val_acc, pre_max_acc = pre_train(config)

        # logging pretrain data into wandb
        wandb.log({"pre_epoch": pre_epoch,"pre_train/train_loss": pre_train_loss,
        "pre_train/train_acc": pre_train_acc,"pre_val/val_loss": pre_val_loss,
        "pre_val/val_acc": pre_val_acc, "pre_max acc":pre_max_acc})

      elif mode == 'finetune':
          # Load meta-train set
          print("Preparing dataset loader")
          trainset = LoadmyMUPSftdata('train', config['test'][0], config) 
          train_sampler = CategoriesSampler(trainset.label, config['num_batch'], config['way'], config['shot'] + config['train_query'])
          train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=2, pin_memory=True)

          # Load meta-val set
          valset = LoadmyMUPSftdata('val', config['test'][0], config)
          val_sampler = CategoriesSampler(valset.label, 20, config['way'], config['shot'] + config['val_query'])
          val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=2, pin_memory=True)
          
          # Build meta-transfer learning model
          model = MtlLearner(config)

          # Set optimizer 
          optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters())}, \
              {'params': model.base_learner.parameters(), 'lr': config['meta_lr2']}], lr=config['meta_lr1'])
          # Set learning rate scheduler 
          lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])        

          # load pretrained model without FC classifier
          model_dict = model.state_dict()
          method = 'pretrain'
          str_value = config['chosen_val']
          sub = config['test'][0]
          save_path = f'./models/X{sub}/{method}/'
          pretrained_dict = torch.load(osp.join(save_path, f'X{sub}_max_acc_legs_pt_{str_value}' + '.pth'))['params']
          pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
          pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
          model_dict.update(pretrained_dict)
          model.load_state_dict(model_dict)    

          # Set model to GPU
          if torch.cuda.is_available():
              torch.backends.cudnn.benchmark = True
              model = model.cuda()
          
          # Set the meta-train log
          trlog = {}
          max_dict = {}
          #trlog['config'] = vars(config)
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

          # Generate the labels for train set of the episodes
          label_shot = torch.arange(config['way']).repeat(config['shot'])
          if torch.cuda.is_available():
              label_shot = label_shot.type(torch.cuda.LongTensor)
          else:
              label_shot = label_shot.type(torch.LongTensor)
          
          # early stopping to avoid overfitting of data - 7 epochs in question
          early_stopping = utils.EarlyStopping()

          # Start meta-train
          # start_time = time.time()
          for epoch in range(1, config['max_epoch'] + 1):
              start_time = time.time()

              # Set the model to train mode
              model.train()
              # Set averager classes to record training losses and accuracies
              train_loss_averager = utils.Averager()
              train_acc_averager = utils.Averager()

              # Generate the labels for test set of the episodes during meta-train updates
              label = torch.arange(config['way']).repeat(config['train_query'])
              if torch.cuda.is_available():
                  label = label.type(torch.cuda.LongTensor)
              else:
                  label = label.type(torch.LongTensor)

              # Using tqdm to read samples from train loader
              tqdm_gen = tqdm.tqdm(train_loader)
              for i, batch in enumerate(tqdm_gen, 1):
                  # Update global count number 
                  global_count = global_count + 1
                  if torch.cuda.is_available():
                      data, _ = [_.cuda() for _ in batch]
                  else:
                      data = batch[0]
                  p = config['shot'] * config['way']
                  data_shot, data_query = data[:p], data[p:]
                  # Output logits for model
                  logits = model((data_shot, label_shot, data_query, None)) # modified
                  # Calculate meta-train loss
                  loss = F.cross_entropy(logits, label)
                  # Calculate meta-train accuracy
                  acc = utils.count_acc(logits, label)

                  # Add loss and accuracy for the averagers
                  train_loss_averager.add(loss.item())
                  train_acc_averager.add(acc)

                  # Loss backwards and optimizer updates
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

              # Update learning rate
              lr_scheduler.step()

              # Update the averagers
              train_loss_averager = train_loss_averager.item()
              train_acc_averager = train_acc_averager.item()

              print("--- %s seconds ---" % (time.time() - start_time))
              # Start validation for this epoch, set model to eval mode
              model.eval()

              # Set averager classes to record validation losses and accuracies
              val_loss_averager = utils.Averager()
              val_acc_averager = utils.Averager()

              # Generate the labels for test set of the episodes during meta-val for this epoch
              label = torch.arange(config['way']).repeat(config['val_query'])
              if torch.cuda.is_available():
                  label = label.type(torch.cuda.LongTensor)
              else:
                  label = label.type(torch.LongTensor)
                  
              # Print previous information
              if epoch % 10 == 0:
                  print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
              # Run meta-validation
              for i, batch in enumerate(val_loader, 1):
                  if torch.cuda.is_available():
                      data, _ = [_.cuda() for _ in batch]
                  else:
                      data = batch[0]
                  p = config['shot'] * config['way']
                  data_shot, data_query = data[:p], data[p:]
                  #logits = model((data_shot, label_shot, data_query, None))
                  logits = model((data_shot, label_shot, data_query,None))
                  loss = F.cross_entropy(logits, label)
                  acc = utils.count_acc(logits, label)

                  val_loss_averager.add(loss.item())
                  val_acc_averager.add(acc)

              # Update validation averagers
              val_loss_averager = val_loss_averager.item()
              val_acc_averager = val_acc_averager.item()
          
              # Print loss and accuracy for this epoch
              print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))
              
              # Update best saved model
              if val_acc_averager > trlog['max_acc']:
                  trlog['max_acc'] = val_acc_averager
                  trlog['max_acc_epoch'] = epoch
                  save_model(model,config['test'][0],'finetune',config['chosen_val'],mode='mt')
                  max_dict = dict(params=model.encoder.state_dict())
                  meta_model_params_dict = dict(params=model.encoder.state_dict())

              # Update the logs
              trlog['train_loss'].append(train_loss_averager)
              trlog['train_acc'].append(train_acc_averager)
              trlog['val_loss'].append(val_loss_averager)
              trlog['val_acc'].append(val_acc_averager)

              wandb.log({"epoch": epoch,
              "train/train_loss": train_loss_averager,
              "train/train_acc": train_acc_averager,
              "val/val_loss": val_loss_averager,
              "val/val_acc": val_acc_averager})

              early_stopping(val_loss_averager)
              if early_stopping.early_stop:
                    break

          wandb.log({"max acc":trlog['max_acc']})
          evalmodel_dict = model.state_dict()
          max_dict = max_dict['params']
          max_dict = {'encoder.'+k: v for k, v in max_dict.items()}
          max_dict = {k: v for k, v in max_dict.items() if k in evalmodel_dict}
          evalmodel_dict.update(max_dict)

          # model eval mode
          model.load_state_dict(evalmodel_dict)
          if torch.cuda.is_available():
            model.cuda()
          model.eval()
          #model.mode = 'test'

          # Load meta-test set
          test_set = LoadmyMUPSftdata('test',config['test'][0], config)
          split = 40
          sampled_test_data = test_set.data[split:]
          sampled_test_labels = test_set.label[split:]
          test_set.data = test_set.data[:split]
          test_set.label = test_set.label[:split] 
          test_sampler = CategoriesSampler(test_set.label, 20, config['way'], config['shot'] + config['val_query'])
          test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=2, pin_memory=True)

          print('Pre test length:',test_set.data.shape, test_set.label.shape )
          print('Actual test length :', sampled_test_data.shape, sampled_test_labels.shape)
          # Set test accuracy recorder
          test_acc_record = np.zeros((20,))
          test_f1_record = np.zeros((20,))
          test_auc_record=np.zeros((20,))

          # Set accuracy averager
          ave_acc = utils.Averager()

          # Generate labels
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

          Y=label.data.cpu().numpy()            
          # Start meta-test
          
          for i, batch in enumerate(test_loader, 1):
              if torch.cuda.is_available():
                  data, _ = [_.cuda() for _ in batch]
              else:
                  data = batch[0]
              k = config['way'] * config['shot'] # 2*20
              data_shot, data_query = data[:k], data[k:]
              logits, last_layer_params = model((data_shot, label_shot, data_query, 'test'))
              acc = utils.count_acc(logits, label)
              logits = logits.data.cpu().numpy()
              predicted=np.argmax(logits, axis=1)
              f1 = f1_score(Y,predicted, average='macro') 
              auc=utils.multiclass_roc_auc_score(Y, predicted)
              ave_acc.add(acc)
              test_acc_record[i-1] = acc
              test_f1_record[i-1] =f1
              test_auc_record[i-1]=auc

              if i % 100 == 0:
                  print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))


          # Calculate the confidence interval, update the logs
          pre_model_accuracy, pm = utils.compute_confidence_interval(test_acc_record)
          f1_m, f1_pm= utils.compute_confidence_interval(test_f1_record)
          auc_m, auc_pm= utils.compute_confidence_interval(test_auc_record)

          #print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
          # print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
          # print('Test f1 {:.4f} + {:.4f}'.format(f1_m, f1_pm))
          # print('Test auc {:.4f} + {:.4f}'.format(auc_m, auc_pm))
          
          wandb.log({"test/test_acc": pre_model_accuracy,
                    "test/test_f1": f1_m,
                    "test/test_auc": auc_m})
          
          final_model_params = meta_model_params_dict['params']
          final_model_params['fc2.weight'] = last_layer_params ['fc2_w']
          final_model_params['fc2.bias'] = last_layer_params ['fc2_b']

          print('Pre Test Accuracy :', pre_model_accuracy)
          net = MindReader()
          net.load_state_dict(final_model_params)
          #net = net.float()
          net.eval()
          ftest_X = sampled_test_data
          ftest_Y = sampled_test_labels

          ftest_X = torch.from_numpy(ftest_X)
          preds = []

          for sample in ftest_X:
            sample = sample[np.newaxis, :, :, :]
            logits = net(sample)
            logits = logits.data.cpu().numpy()
            predicted=np.argmax(logits, axis=1)
            prediction = predicted[-1]
            preds.append(prediction)

          preds = np.array(preds)
          print('Actual Test Accuracy :',accuracy_score(ftest_Y, preds))
          wandb.log({"Actual test_acc": accuracy_score(ftest_Y, preds)})

          return final_model_params, pre_model_accuracy, accuracy_score(ftest_Y, preds)