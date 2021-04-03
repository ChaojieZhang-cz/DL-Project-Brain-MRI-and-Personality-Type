import nibabel as nib
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device('cuda')

print('model5_test5')
class hcp_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        subject_name = self.df.iloc[idx]['Subject']
        image_path ='../../data/hcp2/'+str(subject_name)+'/T1w/T1w_acpc_dc_restore_brain.nii.gz'
        image = nib.load(image_path)
        image_array = image.get_fdata()
        
        #Normaliztion
        image_array = (image_array - image_array.mean()) / image_array.std()
        
        if self.train:# rotate the image (data augementation)
            rand_num = np.random.random()
            if rand_num > 0.7:
                pass # not finish yet
        
        #label = self.df.loc[idx][['N','E','O','A','C']].values.astype(int)
        label = self.df.loc[idx][['C']].values[0].astype(int) # predict C
        
        sample = {'x': image_array[None,:], 'y': label}
        
        return sample
        
        
train_df_path = '../../train.csv'
val_df_path = '../../val.csv'
test_df_path = '../../test.csv'
transformed_dataset = {'train': hcp_dataset(train_df_path, train = True),
                       'validate':hcp_dataset(val_df_path),
                       'test':hcp_dataset(test_df_path),}
bs = 2
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}



class CNN_3D(nn.Module):
    def __init__(self):
        super(CNN_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.conv2 = nn.Conv3d(8, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.conv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.conv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.activation4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.conv5 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.activation5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.fc1 = nn.Linear(64,1000)
        self.fc2 = nn.Linear(1000,3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation4(x)
        x = self.maxpool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)
        x = self.maxpool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
model = CNN_3D()
model = model.to(device)
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight,gain=1.0)

def train_model(model, dataloader, optimizer, loss_fn, num_epochs = 10,\
                acc_dict = None, loss_dict = None, best_acc = None,verbose = True, scheduler=None):
    if acc_dict == None:
        acc_dict = {'train':[],'validate':[]}
    if loss_dict == None:
        loss_dict = {'train':[],'validate':[]}
    if best_acc == None:
        best_acc = 0
    phases = ['train','validate']
    since = time.time()
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
                image = data['x'].to(device,dtype=torch.float)
                label = data['y'].to(device,dtype=torch.long)
                output = model(image)
                loss = loss_fn(output, label)
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
                save_model(best_model_wts, model, acc_dict, loss_dict, best_acc)#####
            else:
                if scheduler:
                    scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    
    return model, acc_dict, loss_dict

def save_model(best_model_wts, model, acc_dict, loss_dict, best_acc):
    state = {'best_model_wts':best_model_wts, 'model':model, 'acc_dict':acc_dict,\
             'loss_dict':loss_dict, 'best_acc':best_acc}
    torch.save(state, 'checkpoint_Model5_test5.pt')
    return None


       
lr_rate = 0.01
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_rate)
train_model(model, dataloader, optimizer, loss_fn, num_epochs = 100, verbose = True, scheduler=None)

