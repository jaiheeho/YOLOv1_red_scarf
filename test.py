import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import vocModel.nntools as nt
import vocData as voc
import vocModel
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", default='data/model_1', help="output directory of model to be saved",\
                    type=str)

parser.add_argument("-e","--epoch", default=5, help="number of epoch to train",
                    type=int)

args = parser.parse_args()

output_dir = args.output_dir
num_epoch = args.epoch

print (output_dir, num_epoch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset_root_dir = "/datasets/ee285f-public/PascalVOC2012/"

train_set = voc.VOCDetection_Yolo(dataset_root_dir,  image_set = 'train')
val_set = voc.VOCDetection_Yolo(dataset_root_dir, image_set = 'val')

if __name__ == '__main__':
    net = vocModel.YoloNet(7, 2, 20, 5,0.5)
    net = net.to(device)
    lr = 1e-5
    adam = torch.optim.Adam(net.parameters(), lr=lr)
    stats_manager = vocModel.DetectionStatsManager()
    output_dir = args.output_dir
    num_epoch = args.epoch
    exp1 = nt.Experiment(net, train_set, val_set, adam, stats_manager,batch_size=4,
                         output_dir=output_dir, perform_validation_during_training=False)
    exp1.run(num_epochs=num_epoch)