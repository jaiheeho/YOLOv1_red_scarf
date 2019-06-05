import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision as tv
from torchvision import models

import vocModel.nntools as nt
import math
from vocModel.utils import cal_iou

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class YOLOObjectDetector(nt.NeuralNetwork):
    def __init__(self, S,B,l_coord,l_noobj):
        super(YOLOObjectDetector, self).__init__()
        self.yololoss = yoloLoss(S,B,l_coord,l_noobj)
    def criterion(self, y, d):
        return self.yololoss(y, d)



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1470):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        self.avgpool = nn.AvgPool2d(2) #fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_detnet_layer(self,in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        
#         x = F.sigmoid(x)
        
        x = torch.sigmoid(x)
        # x = x.view(-1,7,7,30)
        x = x.permute(0,2,3,1) #(-1,7,7,30)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class YoloNet_res(YOLOObjectDetector):
    def __init__(self, S, B, C, l_coord,l_noobj):
        super(YoloNet_res, self).__init__(S, B, l_coord,l_noobj)
    
        self.S = S
        self.B = B
        self.C = C
        
        self.net = resnet50()
        resnet = models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = self.net.state_dict()
        for k in new_state_dict.keys():
#             print(k)
            if k in dd.keys() and not k.startswith('fc'):
#                 print('yes')
                dd[k] = new_state_dict[k]
        self.net.load_state_dict(dd)
   
    def forward(self, x):
        return self.net.forward(x)
        
    
class YoloNet(YOLOObjectDetector):
    def __init__(self, S, B, C, l_coord,l_noobj):
        super(YoloNet, self).__init__(S, B, l_coord,l_noobj)

        self.S = S
        self.B = B
        self.C = C
        
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.features = vgg.features
        
        self.red1 = nn.Sequential(
            ReductionLayer(512, 512, 1024),
            ReductionLayer(1024, 512, 1024),
        )

        self.conv1 = BasicConv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv4 = BasicConv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(4096, (B*5+C) * S * S),
            nn.Sigmoid()
        )
        
        self.classifier[0].weight.data.normal_(0, 0.01)
        self.classifier[0].bias.data.zero_()
        self.classifier[3].weight.data.normal_(0, 0.01)
        self.classifier[3].bias.data.zero_()        


    def forward(self, x):
        
        n = x.size()[0]
        
        x = self.features(x)
        x = self.red1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        
        output = self.classifier(x)
        
        S = self.S
        B = self.B
        C = self.C
        output = output.view(n, S, S, B*5+C)
        return output
        
class YoloNet_IMAGENET(nt.NeuralNetwork):
    def __init__(self):
        self._initialize_weights()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.red3 = nn.Sequential(
            ReductionLayer(192, 128, 256),
            ReductionLayer(256, 256, 512)
        )
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.red4 = nn.Sequential(
            ReductionLayer(512, 256, 512),
            ReductionLayer(512, 256, 512),
            ReductionLayer(512, 256, 512),
            ReductionLayer(512, 256, 512),
            ReductionLayer(512, 512, 1024)
        )
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.red5 = nn.Sequential(
            ReductionLayer(1024, 512, 1024),
            ReductionLayer(1024, 512, 1024),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, 1000)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.red3(x)
        x = self.maxpool3(x)
        x = self.red4(x)
        x = self.maxpool4(x)
        x = self.red5(x)
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord = 5.0,l_noobj = 0.5):
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        
        
#     def compute_iou(self, box1, box2):
#         '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
#         Args:
#           box1: (tensor) bounding boxes, sized [N,4].
#           box2: (tensor) bounding boxes, sized [M,4].
#         Return:
#           (tensor) iou, sized [N,M].
#         '''
#         N = box1.size(0)
#         M = box2.size(0)

#         lt = torch.max(
#             box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
#             box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
#         )

#         rb = torch.min(
#             box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
#             box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
#         )

#         wh = rb - lt  # [N,M,2]
#         wh[wh<0] = 0  # clip at 0
#         inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

#         area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
#         area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
#         area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
#         area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

#         iou = inter / (area1 + area2 - inter)
#         return iou
    
#     def forward(self, pred, target):
#         lamb_class_prob = 1
#         lamb_obj_conf = 1
#         lamb_noobj_conf = self.l_noobj
#         lamb_coord = self.l_coord
        
#         class_loss = 0
#         conf_loss = 0 
#         local_loss = 0

#         n = pred.size()[0]

#         # find where iamge is present of the grid
#         object_mask = target[:,:,:,4] > 0
#         object_mask = object_mask.unsqueeze(-1).expand_as(target)
#         noobject_mask = 1-object_mask

#         ## separate output and prediction 
#         object_pred = (pred[object_mask]).contiguous().view(-1,30)
#         object_target = (target[object_mask]).contiguous().view(-1,30)

#         noobj_pred = (pred[noobject_mask]).contiguous().view(-1,30)
#         noobj_target = (target[noobject_mask]).contiguous().view(-1,30)

#         # 1. Classification loss for both obj and no_boj
#         object_pred_class = object_pred[:,10:]
#         object_target_class = object_target[:,10:]
#         class_loss += lamb_class_prob*F.mse_loss(object_pred_class,object_target_class, reduction='sum')

#         # 2. Confidence loss for no_obj
#         noobj_pred_conf = noobj_pred[:,[4,9]]
#         noobj_target_conf = noobj_target[:,[4,9]]
#         conf_loss += lamb_noobj_conf * F.mse_loss(noobj_pred_conf,noobj_target_conf, reduction ='sum')

#         # 3. Loss for grids with object
#         object_pred_coord = torch.cat((object_pred[:,:4],object_pred[:,5:9]), dim =1).contiguous().view(object_pred.size()[0]*2,-1)
#         object_target_coord = torch.cat((object_target[:,:4],object_target[:,5:9]), dim =1).contiguous().view(object_target.size()[0]*2,-1)

#         object_pred_coord_xy = torch.cat([object_pred_coord[:,:2]/7.0 - torch.pow(object_pred_coord[:,2:],2)/2.0,\
#                                           object_pred_coord[:,:2]/7.0 + torch.pow(object_pred_coord[:,2:],2)/2.0],\
#                                          dim=1)
#         object_target_coord_xy = torch.cat([object_target_coord[:,:2]/7.0 - torch.pow(object_target_coord[:,2:],2)/2.0,\
#                                           object_target_coord[:,:2]/7.0 + torch.pow(object_target_coord[:,2:],2)/2.0],\
#                                          dim=1)

#         # compute iou to decide responsible bounding box. 
#         ious = cal_iou(object_pred_coord_xy, object_target_coord_xy)
#         ious = ious[range(ious.size()[0]),range(ious.size()[0])].view(-1,2)
#         best_boxes = torch.eq(ious, torch.max(ious, 1)[0].unsqueeze(1)).view(-1,1)
#         ious = ious.view(-1,1)

#         # 3-1Calculated localization error for selected bounding boxes.
#         selected_pred_coord = object_pred_coord*best_boxes.expand_as(object_pred_coord).float()
#         selected_target_coord = object_target_coord*best_boxes.expand_as(object_target_coord).float()
#         local_loss = lamb_coord * F.mse_loss(selected_pred_coord, selected_target_coord, reduction='sum')

#         # 3-1Calculat confidence error for selected bounding boxes.
#         object_pred_conf = object_pred[:,[4,9]].view(-1,1)
#         selected_pred_conf = object_pred_conf*best_boxes.float()
# #         selected_target_conf = ious.expand_as(selected_pred_conf)*best_boxes.float()
#         selected_target_conf = ious*best_boxes.float()
#         conf_loss += lamb_obj_conf * F.mse_loss(selected_pred_conf,selected_target_conf,reduction='sum')

#         # 3-2Calculat confidence error for unselected bounding boxes.
#         object_target_conf = object_target[:,[4,9]].view(-1,1)
#         unselected_pred_conf = object_pred_conf*(1-best_boxes).float()
#         unselected_target_conf = object_target_conf * (1-best_boxes).float()
#         conf_loss += lamb_noobj_conf * F.mse_loss(unselected_pred_conf, unselected_target_conf, reduction='sum')

#         loss = local_loss + conf_loss + class_loss
#         loss = loss/n
#         return loss

    
    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum')

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = torch.FloatTensor(box1.size())
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
#             iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            iou = cal_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]

            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = box_target_iou.cuda()
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum') +\
        F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],reduction='sum')

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,reduction='sum')

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N
                 
    
class ReductionLayer(nn.Module):
    def __init__(self, in_channels, ch3x3red, ch3x3):
        super(ReductionLayer, self).__init__()

        self.red = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        output = self.red(x)
        return output

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        import scipy.stats as stats
        X = stats.truncnorm(-2, 2, scale=0.01)
        values = torch.as_tensor(X.rvs(self.conv.weight.numel()), dtype=self.conv.weight.dtype)
        values = values.view(self.conv.weight.size())
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)
#         return F.rrelu(x, inplace=True)    


