import torch
import os.path as osp

from mups.meta_finetuner import meta_finetune
import src.meta_utils as utils

# Set the GPU id
if torch.cuda.is_available(): 
  print('Using device :', torch.cuda.get_device_name(torch.cuda.current_device()))

seed = 42
# Set manual seed for PyTorch
if seed == 0:
    print('Using random seed.')
    torch.backends.cudnn.benchmark = True
else:
    print('Using manual seed:', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for pretraining alone
subj_list = [k for k in range(1,10)]
test_subject = 1 #update subject here
val_list = [4,5,6] # have your list of val subjects here
modes = ['pretrain','finetune']
param_dict_list = []
for val in val_list:
#val = 4
  for mode in modes:
    #mode = 'finetune' # run pretrain first
    #mode = 'pretrain' # then run with mode finetune

    if test_subject in range(1,10):
        print('Old subject')
        train_list = list(set(subj_list) - set([test_subject]))
        #val = random.choice(train_list)
        train_list = list(set(train_list)- set([val]))
    else :
        print('New subject')
        #val = random.choice(subj_list)
        train_list = list(set(subj_list) - set([val]))  

    print(f"Running model for test subject {test_subject} with validation {val}")
    if mode == 'pretrain':
        config = {'clstype' :'legs','max_epoch': 2,'pre_max_epoch': 5,'seed':42,'chosen_val':str(val),#'test_val': x, #'train_size': 4,'val_size': 4
            "pre_lr": 0.05,"pre_gamma": 0.3,"update_step":20,"pre_step_size":5,"pre_batch_size":18,"gamma":0.3,
            "meta_lr1":0.0001,"meta_lr2":0.001,"num_batch":12,"step_size":2,'shot':5,'way':2,'train_query':1,'val_query':1,
            'base_lr':0.00001,"train":train_list,"val":[val],"test":[test_subject],'embed_size':200}
        meta_finetune(config, mode=mode)
    else :
        for x in range(0,5):
            config = {'clstype' :'legs','max_epoch': 2,'pre_max_epoch': 5,'seed':42,'chosen_val':str(val),'test_val': x, #'train_size': 4,'val_size': 4
                    "pre_lr": 0.05,"pre_gamma": 0.3,"update_step":20,"pre_step_size":5,"pre_batch_size":18,"gamma":0.3,
                    "meta_lr1":0.0001,"meta_lr2":0.001,"num_batch":12,"step_size":2,'shot':5,'way':2,'train_query':1,'val_query':1,
                    'base_lr':0.00001,"train":train_list,"val":[val],"test":[test_subject],'embed_size':200}
            print('For val trial :', x)
            final_model_params, pretest_acc , test_acc = meta_finetune(config, mode=mode)
            param_dict_list.append(((final_model_params,pretest_acc),test_acc))
        current_mode = mode

if current_mode == 'finetune':
    final_model_params, max = utils.GetBestDict(param_dict_list)
    # take actual test accuracy into account while selecting model
    # bestModels = utils.Sort_Tuple(dict_list)[:3] 
    # getting top best 3 models here --> maybe for majority voting classifier if we get three models with >0.75 accuracy
    # final model is saved here
    if max > 0.68 :
        sub = utils.subName(config['test'][0])
        save_path = f'./closedloop/models/{sub}/'
        torch.save(final_model_params, osp.join(save_path, f'{sub}_metamodel' + '.pth'))
        print('Model saved chico <3')
    else :
        print('Low Accuracy Chico')

    print('Model finetuned compeleted')