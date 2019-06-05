import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import vocModel.nntools as nt
import vocData as voc
import vocModel
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset_root_dir = "/datasets/ee285f-public/PascalVOC2012/"
lr = 1e-3
batch_size = 8

def voc_data_loader():
    dataset = voc.VOCDetection(dataset_root_dir,  image_set = 'trainval')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = 0.1
    test_split = 0.1
    val_split = int(np.floor(validation_split * dataset_size))
    test_split = int(np.floor(test_split * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[val_split+test_split:],\
    indices[:val_split], indices[val_split:val_split+test_split]

    print (len(train_indices) + len(val_indices) + len(test_indices))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler =  SubsetRandomSampler(test_indices)
    # Define data loaders
    print (batch_size)
    train_loader = td.DataLoader(dataset, batch_size=batch_size, \
                                 sampler=train_sampler,\
                                 drop_last=True, pin_memory=True)

    val_loader = td.DataLoader(dataset, batch_size=batch_size,\
                               sampler=valid_sampler, \
                               drop_last=True, pin_memory=True)

    test_loader = td.DataLoader(dataset, batch_size=1,\
                               sampler=test_sampler, \
                               drop_last=False, pin_memory=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = voc_data_loader()
print ("Dataset is divided into train :{}, val : {}, test :{} ".format(len(train_loader), len(val_loader), len(test_loader)))
# net = vocModel.YoloNet(7, 2, 20, 5,0.5)
net = vocModel.YoloNet_res(7, 2, 20, 5,0.5)
net = net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 5*1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
stats_manager = vocModel.DetectionStatsManager()
exp1 = nt.Experiment(net, train_loader, val_loader, optimizer, stats_manager,batch_size=batch_size,
                     output_dir="data/resnet_3", perform_validation_during_training=True)

exp1.run(num_epochs=100)
