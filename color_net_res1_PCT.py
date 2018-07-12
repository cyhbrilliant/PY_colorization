import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import vgg19_trainable as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
'''
没有指导
只有10层残差
nobn
'''
H=256
W=256

# imgpath='E:/chengzirui/dataset/images256/'
#
# record_list = []
# input_file = open('data_list/places_train.txt', 'r')
# for line in input_file:
#     line = line.strip()
#     record_list.append(line)
# record_list=np.array(record_list)
file = h5py.File('H5color.h5','r')
# file = h5py.File('H5color_imagenet.h5','r')
InputAll = file['Input'][:]

def getBatch(Batch_num):
    Input =[]
    Label=[]
    for i in range(Batch_num):
        index=np.random.randint(0,InputAll.shape[0])
        # index = np.random.randint(0, record_list.shape[0])
        img_testac=InputAll[index, :, :, :]
        img_rgb=cv2.cvtColor(img_testac,cv2.COLOR_LAB2RGB)
        img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
        Input.append(img_gray)
        Label.append(img_rgb)

    return np.array(Input),np.array(Label)



#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
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

train_mode = tf.placeholder(tf.bool)
vgg1 = vgg19.Vgg19('./vgg19.npy',False)
vgg2 = vgg19.Vgg19('./vgg19.npy',False)
vgg1.build(OUT, train_mode)
vgg2.build(Ypy,train_mode)

# loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Yp,logits=OUT))
loss=tf.reduce_mean(tf.square(vgg2.conv2_2-vgg1.conv2_2))
TrainStep=tf.train.AdamOptimizer(0.0001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'color_session/session_res_pct.ckpt')


for iter in range(1000000):
    print('\n',iter)
    Batch_num=16
    Input_batch,Label_batch=getBatch(Batch_num)
    error,result=sess.run([loss,TrainStep],feed_dict={Xp:Input_batch,Yp:Label_batch,train_mode:False})
    print(error)
    # if (iter+1)%50==0:
    #     valerror=sess.run([loss],feed_dict={imageDATA:InputAll[10:20,:,:,:]})
    #     print(valerror)
    if (iter+1)%300==0:
        path = saver.save(sess,'color_session/session_res_pct.ckpt')
        print(path)