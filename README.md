# ECE 285 Object Detection Project 
## project c : YOLO 

## 1. Train 
Everything about train is in "yolo_train.ipynb" jupyter notebook file. Learning process code is written very similar to course assignment using nntools.py at the end of file we have code which generate the result images from neural net.


## 2. mAP 
Calculate mean Average Precision is in "mAP.ipynb" jupter notebook file. Running the jupyter code is very straightforward.

## 3. DEMO

## 4. Code

### 4_1 vocData : PASCAL VOC DATA
PASCAL VOC DATA has two sets of files. One set is images for training and one set is annotations of image which include bounding boxes and corresponding class. We separated the codes for data loading and YOLO specific encoding in vocDATA folder. In here, we utilized torch 'vision dataset' code to read the PASCAL VOC data set and used special encoding for YOLO object detection. 

### 4_2 vocMOdel : YOLO Neural Network model
Architecture of YOLO consists of YOLO Neural Network and YOLO cost function. 'vocModel/YOLO.py' includes the YOLO NN written and YOLO loss function both written in Pytorch. Other files related to NN training such as 'nntools.py' also exists in vocMOdel.
