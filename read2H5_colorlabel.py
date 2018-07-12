# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import h5py

imgpath='E:/Public/DataSet/ImageNet/ILSVRC2012/ILSVRC2012_img_train/'

record_list = []
input_file = open('data_list/imagenet_train.txt', 'r')
for line in input_file:
    line = line.strip()
    # print(line)
    record_list.append(line)
record_list=np.array(record_list)
print(record_list.shape)

# print(type(record_list[0]))
Input =[]
Label=[]
for i in range(500):
    print(i)
    imgname,labelnum=record_list[i].split(' ')
    tempimg = cv2.imread(imgpath+imgname)
    tempimg=cv2.resize(tempimg,(256,256))
    tempimg_lab=cv2.cvtColor(tempimg,cv2.COLOR_BGR2LAB)
    Input.append(tempimg_lab)
    Label.append(int(labelnum))

Input=np.array(Input)
Label=np.array(Label)
print('OK')

file = h5py.File('H5color_Labelsmall.h5','w')
file.create_dataset('Input', data = Input)
file.create_dataset('Label', data = Label)