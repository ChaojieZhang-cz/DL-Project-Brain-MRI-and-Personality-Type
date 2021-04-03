from __future__ import print_function
import json
import pickle
import timeit
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
sys.path.append('Utils')


#from .LinearModel import alex_net_complete

import torch.nn as nn
from torch.autograd import Function


class NetWork(nn.Module):

    def __init__(self, in_channel=1,feat_dim=1024,expansion = 4, type_name='conv3x3x3', norm_type = 'Instance'):
        super(NetWork, self).__init__()
        

        self.conv = nn.Sequential()

        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))

        if norm_type == 'Instance':
           self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        else:
           self.conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))
        self.conv.add_module('relu0_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        else:
            self.conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))


        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        
        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
            
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        #self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))
        self.conv.add_module('test_sl',nn.AdaptiveMaxPool3d((1,1,1)))
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(512, 1024))


    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)['state_dict']
        pretrained_dict = {k[6:]: v for k, v in list(pretrained_dict.items()) if k[6:] in model_dict and 'conv3_s1' not in k and 'fc6' not in k and 'fc7' not in k and 'fc8' not in k}

        model_dict.update(pretrained_dict)
        

        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])
        return pretrained_dict.keys()

    def freeze(self, pretrained_dict_keys):
        for name, param in self.named_parameters():
            if name in pretrained_dict_keys:
                param.requires_grad = False
                

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x):

        z = self.conv(x)
        
        # given that the Liu et al. paper considered their three linear layers without nonlinearities in betweem to be two linear layers, we have followed similar formatting
        z = self.fc6(z.view(x.shape[0],-1))
        return z


def weights_init(model):
    if type(model) in [nn.Conv3d,nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class alex_net_complete(nn.Module):
    def __init__(self, image_embedding_model, classifier=None):
        super(alex_net_complete, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.classifier = classifier

    def forward(self, input_image_variable):
        image_embedding = self.image_embedding_model(input_image_variable)
        if self.classifier is None:
            logit_res = image_embedding
        else:
            logit_res = self.classifier(image_embedding)

        return logit_res


#from .classifier import LinearClassifierAlexNet

import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class LinearClassifierAlexNet(nn.Module):
    def __init__(self, in_dim, n_hid=200, n_label=3):
        super(LinearClassifierAlexNet, self).__init__()
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LinearClassifier', nn.Linear(in_dim, n_hid))
        self.classifier.add_module('LinearClassifier2', nn.Linear(n_hid, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)

def prepare_model(arch = 'ours', in_channel=1, 
    feat_dim=128, n_hid_main=200, n_label=3, out_dim = 128, 
    expansion = 8, type_name='conv3x3x3', norm_type = 'Instance'):

    if arch == 'ours':
        image_embeding_model = NetWork(in_channel=in_channel,feat_dim=feat_dim, expansion = expansion, type_name=type_name, norm_type=norm_type)
    else:
        print('Wrong Architecture!')
    # generate the classifier
    classifier = LinearClassifierAlexNet(in_dim=feat_dim, n_hid=n_hid_main, n_label=n_label)
    main_model = alex_net_complete(image_embeding_model, classifier)

    return main_model
            
def build_model(config, input_dim = 3):
    arch = config['model']['arch']
    in_channel = config['model']['input_channel']
    feat_dim = config['model']['feature_dim']
    n_label = config['model']['n_label']
    n_hid_main = config['model']['nhid']
    out_dim = config['adv_model']['out_dim']
    expansion = config['model']['expansion']
    type_name = config['model']['type_name']
    norm_type = config['model']['norm_type']

    main_model = prepare_model(arch = arch, in_channel=in_channel, feat_dim=feat_dim, 
        n_hid_main=n_hid_main,  n_label=n_label, out_dim = out_dim, 
        expansion= expansion, type_name=type_name, norm_type=norm_type)

    if config['training_parameters']['pretrain'] is not None:
        best_model_dir = './CNN_design_for_AD-master/saved_model/'
        pretrained_dict = torch.load(best_model_dir+ config['training_parameters']['pretrain'] + '_model_low_loss.pth.tar',map_location='cpu')['state_dict']
        model_dict = main_model.state_dict()
        # 6 slice removes part of name
        #pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:]in model_dict.keys())}
        pretrained_dict_1 = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:]in model_dict.keys())}
        # remove linear layers (do not initialize with pretrained weights)
        pretrained_dict = {k:v for k,v in pretrained_dict_1.items() if k in list(pretrained_dict_1.keys())[:8]}
    
        model_dict.update(pretrained_dict) 
        print(model_dict.keys())
        main_model.load_state_dict(model_dict)
    return main_model

# not used
class AgeEncoding(nn.Module):
    "Implement the AE function."
    def __init__(self, d_model, dropout, out_dim,max_len=240):
        super(AgeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(d_model,512))
        self.fc6.add_module('lrn0_s1',nn.LayerNorm(512))
        self.fc6.add_module('fc6_s3',nn.Linear(512, out_dim))
        
        
    def forward(self, x, age_id):
        y = torch.autograd.Variable(self.pe[age_id,:], 
                         requires_grad=False)
        y = self.fc6(y)

        x += y
        return self.dropout(x)