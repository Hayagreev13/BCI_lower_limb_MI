from src.pretrain_tl_utils import train_pt
from src.finetune_tl_utils import train_ft
from src.meta_utils import subName

import os
import random

def pretrain():
    tuple_list = [] # contains train, validation pairs of data
    for i in range(1,10): # subjects for training
        for j in range(1,10): # subjects for validation
            if i != j :
                tuple_list.append((i,j))
    random.shuffle(tuple_list)
    #subj_list = [k for k in range(1,10)]
    for sample in tuple_list:
      test_subject = sample[0]
      val_subject = sample[1]
      print(f'Pretraining for subject {test_subject} with validation subject {val_subject}')
      random.seed(sample[0])
      config={
      'batch_size' : 256,
      'epochs': 15,
      'receptive_field': 64, 
      'mean_pool':  8,
      'activation_type':  'elu',
      'network' : 'EEGNET',
      'val_subjects': val_subject,
      'test_subject': test_subject,
      'seed':  42,    
      'learning_rate': 0.001,}
      train_pt(config)

def finetune():
    subject_list = [k for k in range(1,10)]
    for subj in subject_list: # only 1 for testing code
      # 🐝 initialise a wandb run
      test = subj
      sub = subName(subj)
      #     load_path = f'/content/drive/MyDrive/myMUPS/DL_finetune_models/{sub}/{sub}_max_acc_legs_{config.val_subjects}_pt'
      for instance in os.scandir(f"./models/DL_finetune/{sub}/"): 
          print(f'Getting pre-trained model from {instance.path}')
          valsubjects = int(instance.path[-4:-3])
          print(f'Finetuning test subject {subj} with pretrained val subject {valsubjects}')
          trials = [ (test-1)*10 + k for k in range(0,10)]
          #trials = [0,1,2,3,4,5,6,7,8,9]
          random.seed(subj)
          for trial_num in range(3,5):
              total = 5
              all_trial_list = []
              while len(all_trial_list) < total:
                  trial_list = random.sample(trials, len(trials)) 
                  if trial_list not in all_trial_list:    
                      all_trial_list.append(trial_list)
                      train_trials = trial_list[:trial_num]
                      val_trials = trial_list[trial_num]
                      test_trials = trial_list[5:]
                      print(f"{train_trials}, {val_trials}, {test_trials}")
                      config={
                      'batch_size' : 256,
                      'epochs': 20,
                      'receptive_field': 64, 
                      'mean_pool':  8,
                      'activation_type':  'elu',
                      'network' : 'EEGNET',
                      'test_subject': test, #number
                      'val_subjects': valsubjects, #number
                      'train_trials': train_trials,
                      'val_trials': val_trials,
                      'test_trials': test_trials,
                      'trial_num': trial_num,
                      'seed':  42,    
                      'learning_rate': 0.001,
                      'filter_sizing':  8,
                      'D':  2,
                      'dropout': 0.25}
                      train_ft(config)

def main():
    pretrain()
    finetune()

if __name__ == '__main__':
    main()
