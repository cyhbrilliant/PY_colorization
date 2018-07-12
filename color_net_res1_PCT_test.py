import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'



imgpath='E:/chengzirui/dataset/images256/'

record_list = []
# input_file = open('data_list/train_imagenet.txt', 'r')
input_file = open('data_list/places_train.txt', 'r')

for line in input_file:
    line = line.strip()
    record_list.append(line)
record_list=np.array(record_list)



index=3449
# tempimg = cv2.imread(record_list[index])
tempimg = cv2.imread(imgpath+record_list[index])
# tempimg=cv2.imread('images/input4.jpg')
tempimg=cv2.resize(tempimg,(256,256))
H=tempimg.shape[0]
W=tempimg.shape[1]
Label=cv2.cvtColor(tempimg,cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:]
Input=cv2.cvtColor(tempimg,cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]


#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)

Xp=tf.placeholder(tf.float32, [None, None, None, 1])
Yp=tf.placeholder(tf.float32, [None, None, None, 3])
Xpx=tf.image.resize_nearest_neighbor(Xp,size=[int(H/2),int(W/2)])
Ypy=tf.image.resize_nearest_neighbor(Yp,size=[int(H/2),int(W/2)])

Wconv1=weight_variable([3,3,1,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xpx,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)

for reslayer in range(10):
    #conv+relu
    weightRes1 = weight_variable([3,3,64,64])
    biasRes1 = bias_variable([64])
    convRes = tf.nn.relu(tf.nn.conv2d(conv1,weightRes1,strides=[1,1,1,1],padding='SAME') + biasRes1)
    #conv
    weightRes2 = weight_variable([3,3,64,64])
    biasRes2 = bias_variable([64])
    convRes =tf.nn.conv2d(convRes,weightRes2,strides=[1,1,1,1],padding='SAME') + biasRes2
    conv1 = convRes + conv1

Wconv2=weight_variable([3,3,64,3])
Bconv2=bias_variable([3])
OUT=tf.nn.conv2d(conv1,Wconv2,strides=[1,1,1,1],padding='SAME')+Bconv2

OUT_final=tf.image.resize_nearest_neighbor(OUT,size=[H,W])

sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'color_session/session_res_pct.ckpt')
# saver.restore(sess,'color_session/session_res5.ckpt')


imgout=sess.run(OUT_final,feed_dict={Xp:Input,Yp:Label})
imgfinal=cv2.cvtColor(imgout[0,:,:,:],cv2.COLOR_RGB2BGR)
iter=100
cv2.imwrite('OUTPUT/'+str(iter)+'.jpg',imgfinal)