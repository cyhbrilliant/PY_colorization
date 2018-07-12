# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import h5py

imgpath='E:/chengzirui/dataset/images256/'

record_list = []
input_file = open('data_list/places_train.txt', 'r')
for line in input_file:
    line = line.strip()
    record_list.append(line)
record_list=np.array(record_list)
print(record_list.shape)


Input =[]
for i in range(500000):
    print(i)
    tempimg = cv2.imread(imgpath+record_list[i*4])
    tempimg=cv2.resize(tempimg,(256,256))
    # plt.imshow(tempimg)
    # plt.show()
    # tempimg_lab=cv2.cvtColor(tempimg,cv2.COLOR_BGR2LAB)
    Input.append(tempimg)
Input=np.array(Input)
print('OK')

file = h5py.File('placeBGR50W.h5','w')
file.create_dataset('Input', data = Input)