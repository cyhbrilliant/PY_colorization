import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'



# imgpath='E:/Public/DataSet/imagenet_new/ILSVRC2012_img_val/'
imgpath='E:/chengzirui/dataset/images256/'
record_list = []
# input_file = open('data_list/imagenet_val.txt', 'r')
# input_file = open('data_list/places_train.txt', 'r')
input_file = open('data_list/places_val.txt', 'r')

for line in input_file:
    line = line.strip()
    record_list.append(line)
record_list=np.array(record_list)

H=256
W=256


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

imageDATA = tf.placeholder(tf.float32, [None, None, None, 3])
Xp=imageDATA[:,:,:,0:1]/255
Yp=imageDATA[:,:,:,1:3]/255
Xp=tf.image.resize_nearest_neighbor(Xp,size=[int(H/2),int(W/2)])
Yp=tf.image.resize_nearest_neighbor(Yp,size=[int(H/2),int(W/2)])

Wconv1=weight_variable([5,5,1,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xp,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)

for reslayer in range(10):
    #conv+relu
    weightRes1 = weight_variable([3,3,64,64])
    biasRes1 = bias_variable([64])
    convRes = tf.nn.relu(tf.nn.conv2d(conv1,weightRes1,strides=[1,1,1,1],padding='SAME') + biasRes1)
    #conv
    weightRes2 = weight_variable([3,3,64,64])
    biasRes2 = bias_variable([64])
    convRes =tf.nn.conv2d(convRes,weightRes2,strides=[1,1,1,1],padding='SAME') + biasRes2
    conv1 = tf.nn.relu(convRes + conv1)

Wconv2=weight_variable([3,3,64,2])
Bconv2=bias_variable([2])
OUT=tf.nn.sigmoid(tf.nn.conv2d(conv1,Wconv2,strides=[1,1,1,1],padding='SAME')+Bconv2)

# loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Yp,logits=OUT))
loss_mat=Yp-OUT
loss=tf.reduce_mean(tf.square(Yp-OUT))
loss0=tf.reduce_mean(tf.square(Yp-0.5))
OUT_final=tf.image.resize_nearest_neighbor(OUT,size=[H,W])*255

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'color_session/session_res_orin.ckpt')
# saver.restore(sess,'color_session/session_res5.ckpt')



psnr=[]
for index in range(record_list.shape[0]):
    print(str(index/record_list.shape[0]*100)+'%')
    tempimg = cv2.imread(imgpath+record_list[index])
    tempimg=cv2.resize(tempimg,(256,256))
    tempimg_lab=cv2.cvtColor(tempimg,cv2.COLOR_BGR2LAB)[np.newaxis,:,:,:]
    tempimg_lab=tempimg_lab.astype(dtype=np.float32)

    error = sess.run(loss, feed_dict={imageDATA: tempimg_lab})
    errorP=10.*np.log10(1./error)
    print(errorP)
    psnr.append(errorP)


psnr=np.array(psnr)
np.savetxt('psnr_places.csv', psnr, fmt='%s', delimiter=',')
