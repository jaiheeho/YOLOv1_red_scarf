import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import nntools as nt
import vocData as voc
from torch.utils.data import DataLoader
import cv2


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset_root_dir = "/datasets/ee285f-public/PascalVOC2012/"

train_set = voc.VOCDetection_Yolo(dataset_root_dir,  image_set = 'train')
val_set = voc.VOCDetection_Yolo(dataset_root_dir, image_set = 'val')
print (len(train_set), len(val_set))
