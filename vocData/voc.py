import os
import sys
import tarfile
import collections
import torch
import torchvision as tv
import numpy as np
from .vision import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from .utils import download_url, check_integrity
class_dict = {'person' :0, 'bird' : 1, 'cat' : 2, 'cow': 3, 'dog': 4, 'horse' : 5, 'sheep' : 6,
              'aeroplane' :7, 'bicycle' :8, 'boat' :9, 'bus':10, 'car':11, 'motorbike' :12, 'train':13,
              'bottle' :14, 'chair':15, 'diningtable':16, 'pottedplant':17, 'sofa': 18, 'tvmonitor':19}

class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        
        super(VOCDetection, self).__init__(root, transforms = voc_yolo_transforms)

        self.image_set = image_set

        voc_root = root
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))
           
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        img, target = self.transforms(img, target)
        
        return img, target


    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    
# def voc_yolo_transforms(img, target, S=7, B =2, C=20):
#     transform = tv.transforms.Compose([
#         tv.transforms.Resize((448, 448)),
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     ])
    
#     img = transform(img)
#     size = target['annotation']['size']
#     width, height = np.float(size['width']), np.float(size['height'])
    
#     labels = []
#     bboxes = []
    
#     if type(target['annotation']['object']) == dict :
#         obj = target['annotation']['object']
        
#         class_idx = int(class_dict[obj['name']])
#         labels.append(class_idx)
#         bndbox = obj['bndbox']
        
#         xmin, ymin, xmax, ymax = np.float(bndbox['xmin']), np.float(bndbox['ymin']), \
#         np.float(bndbox['xmax']), np.float(bndbox['ymax'])
#         bboxes.append([xmin, ymin, xmax, ymax])
#     else :
#         for obj in target['annotation']['object']:
            
#             class_idx = int(class_dict[obj['name']])
#             labels.append(class_idx)
#             bndbox = obj['bndbox']
            
#             xmin, ymin, xmax, ymax = np.float(bndbox['xmin']), np.float(bndbox['ymin']), \
#             np.float(bndbox['xmax']), np.float(bndbox['ymax'])  
#             bboxes.append([xmin, ymin, xmax, ymax])
            
    
#     bboxes = torch.Tensor(bboxes) / torch.Tensor([width, height, width, height])
#     labels = torch.LongTensor(labels)
    
#     target = encode_target_yolo (bboxes, labels, S, B, C)
#     return img, target


def voc_yolo_transforms(img, target, S=7, B =2, C=20):

    transform = tv.transforms.Compose([
        tv.transforms.Resize((448, 448)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    
    img = transform(img)
    size = target['annotation']['size']
    width, height = np.float(size['width']), np.float(size['height'])
    
    labels = []
    bboxes = []
    
    if type(target['annotation']['object']) == dict :
        obj = target['annotation']['object']
        
        class_idx = int(class_dict[obj['name']])
        labels.append(class_idx)
        bndbox = obj['bndbox']
        
        xmin, ymin, xmax, ymax = np.float(bndbox['xmin']), np.float(bndbox['ymin']), \
        np.float(bndbox['xmax']), np.float(bndbox['ymax'])
        bboxes.append([xmin, ymin, xmax, ymax])
    else :
        for obj in target['annotation']['object']:
            
            class_idx = int(class_dict[obj['name']])
            labels.append(class_idx)
            bndbox = obj['bndbox']
            
            xmin, ymin, xmax, ymax = np.float(bndbox['xmin']), np.float(bndbox['ymin']), \
            np.float(bndbox['xmax']), np.float(bndbox['ymax'])  
            bboxes.append([xmin, ymin, xmax, ymax])
            
    
    bboxes = torch.Tensor(bboxes) / torch.Tensor([width, height, width, height])
    labels = torch.LongTensor(labels)
    
    target = encode_target_yolo (bboxes, labels, S, B, C)
    return img, target
                                           
def encode_target_yolo(bbox, labels, S, B, C):
    n_elements = B * 5 + C
    target = torch.zeros((S, S, n_elements))
    grid_num = S
    target = torch.zeros((grid_num,grid_num,30))
    cell_size = 1./grid_num
    wh = bbox[:,2:]-bbox[:,:2]
    cxcy = (bbox[:,2:]+bbox[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample/cell_size).ceil()-1 #
        xy = ij*cell_size
        delta_xy = (cxcy_sample -xy)/cell_size
        wh[i] = torch.sqrt(wh[i])
        for j in range(B):
            target[int(ij[1]),int(ij[0]), j * 5 : 2 + j * 5] = delta_xy
            target[int(ij[1]),int(ij[0]), 2 + j * 5 : 4 + j * 5] = wh[i]
            target[int(ij[1]),int(ij[0]), 4 + j * 5] = 1
        
        target[int(ij[1]),int(ij[0]),5+(B-1) * 5 : ] = torch.zeros(C)
        target[int(ij[1]),int(ij[0]),int(labels[i])+ 5 + (B-1) * 5] = 1

    return target