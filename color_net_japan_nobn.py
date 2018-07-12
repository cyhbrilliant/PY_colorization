import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
'''
japan
没有扩分布的
没有bn
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
# file = h5py.File('H5color_imagenet.h5','r')
file = h5py.File('H5color_imagenet.h5','r')
InputAll = file['Input'][:]

def getBatch(Batch_num):
    Input =[]
    for i in range(Batch_num):
        index=np.random.randint(0,InputAll.shape[0])
        img_testac=InputAll[index, :, :, :]
        # img_testac[:,:,2]=(((img_testac[:,:,2]-img_testac[:,:,2].min())/(img_testac[:,:,2].max()-img_testac[:,:,2].min())))*255
        # img_testac[:,:,1]=(((img_testac[:,:,1]-img_testac[:,:,1].min())/(img_testac[:,:,1].max()-img_testac[:,:,1].min())))*255
        img_testac[:, :, 0] = img_testac[:, :, 0] - img_testac[:, :, 0].mean()+128
        img_testac[:, :, 1] = img_testac[:, :, 1] - img_testac[:, :, 1].mean()+128
        Input.append(img_testac)
    return np.array(Input)



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
Yp=tf.image.resize_nearest_neighbor(Yp,size=[int(H/2),int(W/2)])

#W_low_level
low_Wconv1=weight_variable([3, 3, 1, 64])
low_Bconv1=bias_variable([64])
low_Wconv2=weight_variable([3, 3, 64, 128])
low_Bconv2=bias_variable([128])
low_Wconv3=weight_variable([3, 3, 128, 128])
low_Bconv3=bias_variable([128])
low_Wconv4=weight_variable([3, 3, 128, 256])
low_Bconv4=bias_variable([256])
low_Wconv5=weight_variable([3, 3, 256, 256])
low_Bconv5=bias_variable([256])
low_Wconv6=weight_variable([3, 3, 256, 512])
low_Bconv6=bias_variable([512])

#low_level1
CH1_conv1=tf.nn.relu(tf.nn.conv2d(Xp, low_Wconv1, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv1)
CH1_conv2=tf.nn.relu(tf.nn.conv2d(CH1_conv1, low_Wconv2, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv2)
CH1_conv3=tf.nn.relu(tf.nn.conv2d(CH1_conv2, low_Wconv3, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv3)
CH1_conv4=tf.nn.relu(tf.nn.conv2d(CH1_conv3, low_Wconv4, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv4)
CH1_conv5=tf.nn.relu(tf.nn.conv2d(CH1_conv4, low_Wconv5, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv5)
CH1_conv6=tf.nn.relu(tf.nn.conv2d(CH1_conv5, low_Wconv6, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv6)

#low_level2
Xp_224=tf.image.resize_nearest_neighbor(Xp,size=[224,224])
CH2_conv1=tf.nn.relu(tf.nn.conv2d(Xp_224, low_Wconv1, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv1)
CH2_conv2=tf.nn.relu(tf.nn.conv2d(CH2_conv1, low_Wconv2, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv2)
CH2_conv3=tf.nn.relu(tf.nn.conv2d(CH2_conv2, low_Wconv3, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv3)
CH2_conv4=tf.nn.relu(tf.nn.conv2d(CH2_conv3, low_Wconv4, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv4)
CH2_conv5=tf.nn.relu(tf.nn.conv2d(CH2_conv4, low_Wconv5, strides=[1, 2, 2, 1], padding='SAME') + low_Bconv5)
CH2_conv6=tf.nn.relu(tf.nn.conv2d(CH2_conv5, low_Wconv6, strides=[1, 1, 1, 1], padding='SAME') + low_Bconv6)


#W_Global
global_Wconv1=weight_variable([3,3,512,512])
global_Bconv1=bias_variable([512])
global_Wconv2=weight_variable([3,3,512,512])
global_Bconv2=bias_variable([512])
global_Wconv3=weight_variable([3,3,512,512])
global_Bconv3=bias_variable([512])
global_Wconv4=weight_variable([3,3,512,512])
global_Bconv4=bias_variable([512])
global_Wfc5=weight_variable([25088,1024])
global_Bfc5=bias_variable([1024])
global_Wfc6=weight_variable([1024,512])
global_Bfc6=bias_variable([512])
global_Wfc7=weight_variable([512,256])
global_Bfc7=bias_variable([256])

#Global
global_conv1=tf.nn.relu(tf.nn.conv2d(CH2_conv6,global_Wconv1,strides=[1,2,2,1],padding='SAME')+global_Bconv1)
global_conv2=tf.nn.relu(tf.nn.conv2d(global_conv1,global_Wconv2,strides=[1,1,1,1],padding='SAME')+global_Bconv2)
global_conv3=tf.nn.relu(tf.nn.conv2d(global_conv2,global_Wconv3,strides=[1,2,2,1],padding='SAME')+global_Bconv3)
global_conv4=tf.nn.relu(tf.nn.conv2d(global_conv3,global_Wconv4,strides=[1,1,1,1],padding='SAME')+global_Bconv4)
global_conv4_line=tf.reshape(global_conv4,shape=[-1,25088])
global_fc5=tf.nn.relu(tf.matmul(global_conv4_line,global_Wfc5)+global_Bfc5)
global_fc6=tf.nn.relu(tf.matmul(global_fc5,global_Wfc6)+global_Bfc6)
global_fc7=tf.nn.relu(tf.matmul(global_fc6,global_Wfc7)+global_Bfc7)


#W_Mid
mid_Wconv1=weight_variable([3,3,512,512])
mid_Bconv1=bias_variable([512])
mid_Wconv2=weight_variable([3,3,512,256])
mid_Bconv2=bias_variable([256])

#Mid
mid_conv1=tf.nn.relu(tf.nn.conv2d(CH1_conv6,mid_Wconv1,strides=[1,1,1,1],padding='SAME')+mid_Bconv1)
mid_conv2=tf.nn.relu(tf.nn.conv2d(mid_conv1,mid_Wconv2,strides=[1,1,1,1],padding='SAME')+mid_Bconv2)



#Fusion
Fu_scaleH=int(H/8)
Fu_scaleW=int(W/8)
FusionL2=tf.reshape(tf.tile(global_fc7,[1,Fu_scaleH*Fu_scaleW]),shape=[-1,Fu_scaleH,Fu_scaleW,256])
Fusion_out=tf.concat((mid_conv2,FusionL2),axis=3)



#W_Colorization
col_Wfu=weight_variable([1,1,512,256])
col_Bfu=bias_variable([256])
col_Wconv1=weight_variable([3,3,256,128])
col_Bconv1=bias_variable([128])
col_Wconv2=weight_variable([3,3,128,64])
col_Bconv2=bias_variable([64])
col_Wconv3=weight_variable([3,3,64,64])
col_Bconv3=bias_variable([64])
col_Wconv4=weight_variable([3,3,64,32])
col_Bconv4=bias_variable([32])
col_Wconv5=weight_variable([3,3,32,2])
col_Bconv5=bias_variable([2])


col_Fusion=tf.nn.relu(tf.nn.conv2d(Fusion_out,col_Wfu,strides=[1,1,1,1],padding='SAME')+col_Bfu)
col_conv1=tf.nn.relu(tf.nn.conv2d(col_Fusion,col_Wconv1,strides=[1,1,1,1],padding='SAME')+col_Bconv1)
upsample1=tf.image.resize_nearest_neighbor(col_conv1,size=[int(H/4),int(W/4)])
col_conv2=tf.nn.relu(tf.nn.conv2d(upsample1,col_Wconv2,strides=[1,1,1,1],padding='SAME')+col_Bconv2)
col_conv3=tf.nn.relu(tf.nn.conv2d(col_conv2,col_Wconv3,strides=[1,1,1,1],padding='SAME')+col_Bconv3)
upsample3=tf.image.resize_nearest_neighbor(col_conv3,size=[int(H/2),int(W/2)])
col_conv4=tf.nn.relu(tf.nn.conv2d(upsample3,col_Wconv4,strides=[1,1,1,1],padding='SAME')+col_Bconv4)
OUT=tf.nn.sigmoid(tf.nn.conv2d(col_conv4,col_Wconv5,strides=[1,1,1,1],padding='SAME')+col_Bconv5)

# loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Yp,logits=OUT))
loss=tf.reduce_mean(tf.square(Yp-OUT))
TrainStep=tf.train.AdamOptimizer(0.0001).minimize(loss)

sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'color_session/session_20.ckpt')


for iter in range(1000000):
    print('\n',iter)
    Batch_num=16
    Input_batch=getBatch(Batch_num)
    error,result=sess.run([loss,TrainStep],feed_dict={imageDATA:Input_batch})
    print(error)
    if (iter+1)%50==0:
        valerror=sess.run([loss],feed_dict={imageDATA:InputAll[10:20,:,:,:]})
        print(valerror)
    if (iter+1)%500==0:
        path = saver.save(sess,'color_session/session_30.ckpt')
        print(path)