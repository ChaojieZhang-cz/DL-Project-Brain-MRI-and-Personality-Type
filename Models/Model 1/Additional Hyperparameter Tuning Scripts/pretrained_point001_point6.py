import os
import yaml
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
import matplotlib.pyplot as plt
import sys

# build model from Liu et al.'s github code

sys.path.insert(1, "./CNN_design_for_AD-master/models/")
import build_model_extrablock


# for untrained use:
#config_name = './CNN_design_for_AD-master/config.yaml'

# pretrained model
config_name = './CNN_design_for_AD-master/config2.yaml'
with open(os.path.join('./'+config_name), 'r') as f:
    cfg = yaml.load(f)

device = torch.device('cuda')

model = build_model_extrablock.build_model(cfg).to(device)

class hcp_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        subject_name = self.df.iloc[idx]['Subject']
        image_path ='./data/hcp2/'+str(subject_name)+'/T1w/T1w_acpc_dc_restore_brain.nii.gz'
        image = nib.load(image_path)
        image_array = image.get_fdata()

        #Normalization
        image_array = (image_array - image_array.mean()) / image_array.std()

        #label = self.df.loc[idx][['N','E','O','A','C']].values.astype(int)
        label = self.df.loc[idx][['N']].values[0].astype(int) # predict C


        sample = {'x': image_array[None,:], 'y': label}

        return sample


bs = 1

# full dataset

train_df_path = './train.csv'
val_df_path = './test.csv'
test_df_path = './val.csv'
transformed_dataset = {'train': hcp_dataset(train_df_path, train = True),
                       'validate':hcp_dataset(val_df_path),
                       'test':hcp_dataset(test_df_path),}

# for debugging and to see if model can learn training set on tiny sample
#sample_df_path = './sample.csv'
#sample_transformed_dataset = {'train': hcp_dataset(sample_df_path, train = True),
#                       'validate':hcp_dataset(sample_df_path),
#                       'test':hcp_dataset(sample_df_path),}
#
#dataloader_sample = {x: DataLoader(sample_transformed_dataset[x], batch_size=bs,
#                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}


# get data_loader
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}


def train_model(model, dataloader, optimizer, loss_fn, interpolation_scale, num_epochs = 10, verbose = True, scheduler=None, output_name="test.txt"):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_acc = 0
    phases = ['train','validate']
    since = time.time()
    number = 0
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0
            if p == 'train':
                model.train()
            else:
                model.eval()

            for data in dataloader[p]:
                optimizer.zero_grad()
                image = F.interpolate(data['x'], mode="trilinear", scale_factor=interpolation_scale)
                image = image.to(device,dtype=torch.float)
                label = data['y'].to(device,dtype=torch.long)
                output = model(image)
                loss = loss_fn(output, label)
                print(number)
                number += 1
                _, preds = torch.max(output, dim = 1)
                num_imgs = image.size()[0]
                running_correct += torch.sum(preds ==label).item()
                running_loss += loss.item()*num_imgs
                running_total += num_imgs
                if p== 'train':
                    loss.backward()
                    optimizer.step()
            epoch_acc = float(running_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            if verbose or (i%10 == 0):
                print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_acc))

            acc_dict[p].append(epoch_acc)
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                save_model(best_model_wts, model, acc_dict, loss_dict)
            else:
                if scheduler:
                    scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, acc_dict, loss_dict

def save_model(best_model_wts, model, acc_dict, loss_dict):
    model_saved = {'best_model_wts':best_model_wts, 'model':model, 'acc_dict':acc_dict, 'loss_dict':loss_dict}
    f=open(output_name,'wb')
    pickle.dump(model_saved,f)
    f.close()
    return None

# from Liu et al.
lr_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_rate)
interpolation_scale = 0.6

output_name = "pretrained_point001_point6.txt"
model, acc_dict, loss_dict = train_model(model, dataloader, optimizer, loss_fn, interpolation_scale, num_epochs = 50, verbose = True, scheduler= MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), output_name = "")
