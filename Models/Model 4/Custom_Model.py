import nibabel as nib
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
import time
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

device = torch.device('cuda')

print('Custom Model Test 5')
print('batch size = 1')


def Random3DCrop(images):
    Crop1 = random.randrange(0,11)
    Crop2 = random.randrange(0,12)
    Crop3 = random.randrange(0,11)
    images = images[Crop1:Crop1+250,Crop2:Crop2+300:,Crop3:Crop3+250]
    return images
    
    
class hcp_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        subject_name = self.df.iloc[idx]['Subject']
        image_path ='/scratch/cz2064/myjupyter/DeepLearning/Final_Project/data/hcp2/'+str(subject_name)+'/T1w/T1w_acpc_dc_restore_brain.nii.gz'
        image = nib.load(image_path)
        image_array = image.get_fdata()
        
        #Preprocess
        image_array = Random3DCrop(image_array)
        image_array = (image_array - image_array.mean()) / image_array.std()
        
        #label = self.df.loc[idx][['N','E','O','A','C']].values.astype(int)
        label = self.df.loc[idx][['C']].values[0].astype(int) # predict C
        
        
        sample = {'x': image_array[None,:], 'y': label}
        
        return sample

train_df_path = '/scratch/cz2064/myjupyter/DeepLearning/Final_Project/train.csv'
val_df_path = '/scratch/cz2064/myjupyter/DeepLearning/Final_Project/val.csv'
test_df_path = '/scratch/cz2064/myjupyter/DeepLearning/Final_Project/test.csv'
transformed_dataset = {'train': hcp_dataset(train_df_path, train = True),
                       'validate':hcp_dataset(val_df_path),
                       'test':hcp_dataset(test_df_path),}
bs = 1
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}

class CNN_3D(nn.Module):
    def __init__(self):
        super(CNN_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        
        self.conv2_1_1 = nn.Conv3d(16,16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv2_2_1 = nn.Conv3d(16,16, kernel_size=(1,1,7), stride=(1,1,1), padding=(0,0,3))
        self.conv2_2_2 = nn.Conv3d(16,16, kernel_size=(1,7,1), stride=(1,1,1), padding=(0,3,0))
        self.conv2_2_3 = nn.Conv3d(16,16, kernel_size=(7,1,1), stride=(1,1,1), padding=(3,0,0))
        self.bn2 = nn.BatchNorm3d(32)
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        self.res1_avgpool = nn.AvgPool3d(3, stride=3)
        self.res1_conv2 = nn.Conv3d(16,32,1,1)
        
        self.conv3_1_1 = nn.Conv3d(32,32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv3_2_1 = nn.Conv3d(32,32, kernel_size=(1,1,7), stride=(1,1,1), padding=(0,0,3))
        self.conv3_2_2 = nn.Conv3d(32,32, kernel_size=(1,7,1), stride=(1,1,1), padding=(0,3,0))
        self.conv3_2_3 = nn.Conv3d(32,32, kernel_size=(7,1,1), stride=(1,1,1), padding=(3,0,0))
        self.bn3 = nn.BatchNorm3d(64)
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
        self.res2_avgpool = nn.AvgPool3d(3, stride=3)
        self.res2_conv3 = nn.Conv3d(32,64,1,1)

        self.conv4_1_1 = nn.Conv3d(64,64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv4_2_1 = nn.Conv3d(64,64, kernel_size=(1,1,7), stride=(1,1,1), padding=(0,0,3))
        self.conv4_2_2 = nn.Conv3d(64,64, kernel_size=(1,7,1), stride=(1,1,1), padding=(0,3,0))
        self.conv4_2_3 = nn.Conv3d(64,64, kernel_size=(7,1,1), stride=(1,1,1), padding=(3,0,0))
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        self.res3_avgpool = nn.AvgPool3d(3, stride=3)
        self.res3_conv4 = nn.Conv3d(64,128,1,1)
        
        self.GobalAveragePooling = nn.AdaptiveMaxPool3d((1,1,1))
        self.fc = nn.Linear(128,3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        residual1 = x
        x = self.maxpool1(x)
        
        
        x1 = self.conv2_1_1(x)
        x2 = self.conv2_1_1(x)
        x2 = self.conv2_2_2(x2)
        x2 = self.conv2_2_3(x2)
        x = torch.cat((x1, x2), 1)
        residual1 = self.res1_avgpool(residual1)
        residual1 = self.res1_conv2(residual1)
        x = x + residual1
        x = self.bn2(x)
        x = self.activation2(x)
        residual2 = x
        x = self.maxpool2(x)
        
        
        x1 = self.conv3_1_1(x)
        x2 = self.conv3_1_1(x)
        x2 = self.conv3_2_2(x2)
        x2 = self.conv3_2_3(x2)
        x = torch.cat((x1, x2), 1)
        residual2 = self.res2_avgpool(residual2)
        residual2 = self.res2_conv3(residual2)
        x = x + residual2
        x = self.bn3(x)
        x = self.activation3(x)
        residual3 = x
        x = self.maxpool3(x)
        
        
        x1 = self.conv4_1_1(x)
        x2 = self.conv4_1_1(x)
        x2 = self.conv4_2_2(x2)
        x2 = self.conv4_2_3(x2)
        x = torch.cat((x1, x2), 1)
        residual3 = self.res3_avgpool(residual3)
        residual3 = self.res3_conv4(residual3)
        x = x + residual3
        x = self.bn4(x)
        x = self.activation4(x)
        
        self.featuremap1 = x.detach()
        x = self.GobalAveragePooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
    torch.save(state, 'checkpoint_Custom_Model_test5.pt')
    return None
    
    
lr_rate = 0.01
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_rate)


train_model(model, dataloader, optimizer, loss_fn, num_epochs = 100, verbose = True, scheduler=None)
