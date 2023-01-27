import os
import os.path as osp
import torch
import src.meta_utils as utils

def save_model(model,sub,method,str_value,mode):
        """The function to save checkpoints.
        Args:
          model : ML model
          sub : subject
          str_value : validation set values
          mode : either pt for pretraining and mt for metatraining
          method : pretrained or finetuned
        """  
        sub = utils.subName(sub)
        save_path = f'./models/'
        if not osp.exists(save_path):
            os.mkdir(save_path)
        save_path = osp.join(save_path, sub)
        if not osp.exists(save_path):
            os.mkdir(save_path)
        save_path = osp.join(save_path, method)
        if not osp.exists(save_path):
            os.mkdir(save_path) 
         
        torch.save(dict(params=model.encoder.state_dict()), osp.join(save_path, f'{str(sub)}_max_acc_legs_{mode}_{str_value}' + '.pth'))
    