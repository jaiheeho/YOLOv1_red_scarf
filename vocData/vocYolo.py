import torchvision as tv
import numpy as np
import torch

from .voc import VOCDetection
class_dict = {'person' :0, 'bird' : 1, 'cat' : 2, 'cow': 3, 'dog': 4, 'horse' : 5, 'sheep' : 6,
              'aeroplane' :7, 'bicycle' :8, 'boat' :9, 'bus':10, 'car':11, 'motorbike' :12, 'train':13,
              'bottle' :14, 'chair':15, 'diningtable':16, 'pottedplant':17, 'sofa': 18, 'tvmonitor':19}

class VOCDetection_Yolo(VOCDetection):
    def __init__(self, root, image_set='train'):
        if (image_set == 'train') : 
            super(VOCDetection_Yolo, self).__init__(root, image_set = image_set,
                                           transforms = voc_yolo_transforms)
        if (image_set == 'val'):
            super(VOCDetection_Yolo, self).__init__(root, image_set = image_set,
                                           transforms = voc_yolo_transforms_test)

        
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

def voc_yolo_transforms_test(img, target, S=7, B =2, C=20):

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

    return img, (bboxes, labels, target)
                      
