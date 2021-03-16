# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# import tensorflow as tf
# import glob # 获取路径
# import tensorflow.keras as keras
# from matplotlib import pyplot as plt # 绘图
# AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
# import os
# import math
# import numpy as np
# from net import resnet_common

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # tf.config.gpu.set_per_process_memory_growth(enabled=True)
# # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# # In[2]:



# img_ori = glob.glob('./1_ch4_training_images/*jpg')

# img_ori

# # In[3]:



# # In[3]:


# img_ori


# # In[4]:


# # label_ori = glob.glob('./2_ch4_training_localization_transcription_gt/*txt')
# score_label = glob.glob('./score_map/*jpg')
# angle_label = glob.glob('./angle_map/*jpg')
# geo_label_0 = glob.glob('./geo_map/*_0.jpg')
# geo_label_1 = glob.glob('./geo_map/*_1.jpg')
# geo_label_2 = glob.glob('./geo_map/*_2.jpg')
# geo_label_3 = glob.glob('./geo_map/*_3.jpg')
# geo_label_4 = glob.glob('./geo_map/*_4.jpg')
# geo_label_5 = glob.glob('./geo_map/*_5.jpg')
# geo_label_6 = glob.glob('./geo_map/*_6.jpg')
# geo_label_7 = glob.glob('./geo_map/*_7.jpg')

# char_to_split = os.sep 

# # In[5]:
# w_init = tf.random_normal_initializer(stddev=0.02)
# gamma_init = tf.random_normal_initializer(1., 0.02)

# def takeFirst(path):
#         # print(path)
#         index = path.split(char_to_split)[2].split('.')[0]
#         return int(index)

# def takeSecond(path):
#         index = path.split(char_to_split)[2].split('_')[0]
#         return int(index)

# def takeThird(path):
#         index = path.split(char_to_split)[2].split('_')[-1].split('.')[0]
#         return int(index)


# # In[6]:

# img_ori.sort(key=takeThird)

# score_label.sort(key=takeFirst)

# angle_label.sort(key=takeFirst)
# geo_label_0.sort(key=takeSecond)
# geo_label_1.sort(key=takeSecond)
# geo_label_2.sort(key=takeSecond)
# geo_label_3.sort(key=takeSecond)
# geo_label_4.sort(key=takeSecond)
# geo_label_5.sort(key=takeSecond)
# geo_label_6.sort(key=takeSecond)
# geo_label_7.sort(key=takeSecond)


import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
from net import resnet_common

os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.gpu.set_per_process_memory_growth(enabled=True)
# tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# In[2]:

char_to_split = os.sep  

img_ori = glob.glob('dataset/Archive/*jpg')


# label_ori = glob.glob('./2_ch4_training_localization_transcription_gt/*txt')
score_label = glob.glob('dataset/my_score_map/*jpg')
angle_label = glob.glob('dataset/my_score_map/*jpg')
geo_label_0 = glob.glob('dataset/my_geo_map/*_0.jpg')
geo_label_1 = glob.glob('dataset/my_geo_map/*_1.jpg')
geo_label_2 = glob.glob('dataset/my_geo_map/*_2.jpg')
geo_label_3 = glob.glob('dataset/my_geo_map/*_3.jpg')
geo_label_4 = glob.glob('dataset/my_geo_map/*_4.jpg')
geo_label_5 = glob.glob('dataset/my_geo_map/*_5.jpg')
geo_label_6 = glob.glob('dataset/my_geo_map/*_6.jpg')
geo_label_7 = glob.glob('dataset/my_geo_map/*_7.jpg')


w_init = tf.random_normal_initializer(stddev=0.02)
gamma_init = tf.random_normal_initializer(1., 0.02)

def takeFirst(path):
        index = path.split(char_to_split)[2].split('.')[0].split('_')[-1]
        print(index)
        return int(index)

def takeSecond(path):
        index = path.split(char_to_split)[2].split('.')[0].split('_')[-1]
        return int(index)

def takeThird(path):
        index = path.split(char_to_split)[2].split('_')[-1].split('.')[0]
        print(index)
        return int(index)


img_ori.sort(key=takeThird)

score_label.sort(key=takeFirst)
angle_label.sort(key=takeFirst)

geo_label_0.sort(key=takeSecond)
geo_label_1.sort(key=takeSecond)
geo_label_2.sort(key=takeSecond)
geo_label_3.sort(key=takeSecond)
geo_label_4.sort(key=takeSecond)
geo_label_5.sort(key=takeSecond)
geo_label_6.sort(key=takeSecond)
geo_label_7.sort(key=takeSecond)


batch_size  = 3
train_img = tf.data.Dataset.from_tensor_slices(img_ori)
label_score_img = tf.data.Dataset.from_tensor_slices(score_label)
label_angle_img = tf.data.Dataset.from_tensor_slices(angle_label)
label_geo_img = tf.data.Dataset.from_tensor_slices((geo_label_0,geo_label_1,geo_label_2,geo_label_3,geo_label_4,geo_label_5,geo_label_6,geo_label_7))


def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img


def read_jpg_gt(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 1)
#     img = tf.image.rgb_to_grayscale(img)
#     img = tf.sum(img,axis=-1)
    print("img.shape",img.shape)
    return img


def load_img_train(path):
    img = read_jpg(path)
    print(path)
#     index = path.split('/')[2].split('.')[0].split('_')[-1]
#     print(path)
    img = tf.image.resize(img,(512,512)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


def load_img_label(path):
    print(path)
    img = read_jpg_gt(path)
    img = tf.image.resize(img,(64,64))
    img = tf.cast(img, tf.float32)/255# - 1#img/127.5-1
    return img


def load_img_label_angle(path):
    # print(path)
    img = read_jpg_gt(path)
    img = tf.image.resize(img,(64,64))
    img = tf.cast(img, tf.float32)/127.5-1# - 1#img/127.5-1
    return img


def load_img_geo_label(path0,path1,path2,path3,path4,path5,path6,path7):
    img_0 = read_jpg_gt(path0)
    img_0 = tf.image.resize(img_0,(64,64)) # resize 会使得在后续显示的时候不是none none
    
    
    img_1 = read_jpg_gt(path1)
    img_1 = tf.image.resize(img_1,(64,64))
    
    
    img_2 = read_jpg_gt(path2)
    img_2 = tf.image.resize(img_2,(64,64))
    
    
    img_3 = read_jpg_gt(path3)
    img_3 = tf.image.resize(img_3,(64,64))
    
    img_4 = read_jpg_gt(path4)
    img_4 = tf.image.resize(img_4,(64,64))
    
    img_5 = read_jpg_gt(path5)
    img_5 = tf.image.resize(img_5,(64,64))
    
    img_6 = read_jpg_gt(path6)
    img_6 = tf.image.resize(img_6,(64,64))
    
    img_7 = read_jpg_gt(path7)
    img_7 = tf.image.resize(img_7,(64,64))
    
    img = tf.stack([img_0, img_1,img_2,img_3,img_4,img_5,img_6,img_7], axis=2)

    return img


# batch_size = 2
train_data = train_img.map(load_img_train,num_parallel_calls = AUTOTUNE).cache().repeat().batch(batch_size)#.shuffle(buffer_size)


buffer_size  = 1

label_score_data = label_score_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().repeat().batch(batch_size)
label_angle_data = label_angle_img.map(load_img_label_angle,num_parallel_calls = AUTOTUNE).cache().repeat().batch(batch_size)
label_geo_data = label_geo_img.map(load_img_geo_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)

train_data_zip = tf.data.Dataset.zip((train_data,label_score_data,label_angle_data,label_geo_data)).shuffle(20)#


def downSample(filter,size,padding ='same',apply_batchnorm=True,w_init_=w_init,gamma_init_=gamma_init):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size,strides = 2,padding=padding,kernel_initializer = w_init_))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization(gamma_initializer=gamma_init_))
        result.add(keras.layers.LeakyReLU())
    return result


def upSample(filter,size,apply_batchnorm=True,w_init_=w_init,gamma_init_=gamma_init):
    result  = keras.Sequential()
    # result.add(keras.layers.Unpooling((2,2),strides = 2,padding = 'same'))
    # result.add(keras.layers.MaxPooling2D((2,2),strides = 0.5,padding = 'same'))
    result.add(keras.layers.Conv2DTranspose(filter,size,strides = 2,padding='same',kernel_initializer =w_init_))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization(gamma_initializer = gamma_init_))
        result.add(keras.layers.LeakyReLU())
    return result


def conv3conv1Sample(filter,size1,size2,apply_batchnorm=True,w_init_=w_init,gamma_init_=gamma_init):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size1,strides = 1,padding='same',kernel_initializer =w_init_))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization(gamma_initializer = gamma_init_))
        result.add(keras.layers.LeakyReLU())
    result.add(keras.layers.Conv2D(filter,size2,strides = 1,padding='same',kernel_initializer =w_init_))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization(gamma_initializer = gamma_init_))
        result.add(keras.layers.LeakyReLU())
    return  result


def afterVGG16():
    inputs = keras.layers.Input(shape=[512,512,3])
    # path = './resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # model_vgg16 = keras.applications.ResNet50V2(weights=None, include_top=False)
    # model_vgg16.load_weights(path, by_name=True,skip_mismatch=True) 
    # model_vgg16.trainable = True
    # x = model_vgg16(inputs)

    resnet50 = resnet_common.ResNet50V2(include_top=False,weights='models/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5',input_tensor=inputs,input_shape=(512,512,3))
    resnet50.trainable = True
    x = resnet50(inputs)
    # inputs = resnet50.input
    # x = resnet50.output

    print('befora first',inputs)


    x = keras.layers.Conv2D(16,7,strides = 2 ,padding='same')(x)
    print('first',x.shape)
    downsample_list = [
        downSample(64,4),
        downSample(128,4),
        downSample(256,4),
        downSample(384,4),
    ]

    downresult_list = []

    i = 0
    for down in downsample_list:
        # print('junejune')
        x = down(x)
        # print('junejune1')
        downresult_list.append(x)
        print('down',i,x.shape)
        i = i+1

    downresult_list = reversed(downresult_list[:-1])#不要最后一层进行反转list

    upsample_list = [
        upSample(256,3),
        upSample(128,3),
        upSample(64,3),
        # upSample(384,3)
    ]
    conv3conv1_list = [
        conv3conv1Sample(128,1,3),
        conv3conv1Sample(64,1,3),
        conv3conv1Sample(32,1,3),
    ]
    i = 0
    for up,inputs_from_down ,conv3conv1 in zip(upsample_list,downresult_list,conv3conv1_list):
        x = up(x)
        print('up',i,x.shape)
        x = keras.layers.Concatenate()([x,inputs_from_down])
        # print('junejune22')
        x = conv3conv1 (x)
        print('up',i,x.shape)
        i = i+1

    pred_score = keras.layers.Conv2D(1,1,strides=1,padding = 'same',activation = 'sigmoid')(x)#,activation = 'sigmoid')(x)
    print(pred_score.shape)
    #     pred_score = 
    angle_map = (keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)-0.5)*np.pi/2
    geo_map_0 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_1 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_2 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_3 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_4 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_5 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_6 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_7 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map = tf.stack([geo_map_0, geo_map_1,geo_map_2,geo_map_3,geo_map_4,geo_map_5,geo_map_6,geo_map_7], axis=3)
    print(geo_map)
    return keras.Model(inputs = inputs,outputs = [pred_score,angle_map,geo_map] )


# net =tf.keras.models.load_model('net.h5')
net = afterVGG16()


def scoreMapLossDice(gt_loss,pred_loss,smooth=1e-5):
    print( gt_loss)
    print( pred_loss)
    batch_size,w,h,c =  gt_loss.shape
    
    gt_loss = tf.reshape(gt_loss,(batch_size,w*h*c))
#     gt_loss_bk = tf.constant([gt_loss])
#     print('gt_loss_bk', gt_loss)
    pred_loss = tf.reshape(pred_loss,(batch_size,w*h*c))
#     print('pred_loss',pred_loss)

    
    intersection = tf.reduce_sum(gt_loss*pred_loss,axis=-1)#
    l = tf.reduce_sum(gt_loss,axis=-1)
#     ll=l.numpy()
#     print('l=',l)
    r = tf.reduce_sum(pred_loss,axis=-1)
#     print('r=',r)
    coeff = (2*intersection+1)/(l+r+1)
#     print('coeff',coeff)

    dice_loss = 1.-tf.reduce_sum(coeff)/batch_size
    return dice_loss*1#,gt_loss,pred_loss


# angle loss
def lossAngle(gt_angle,pred_angle):
    return tf.reduce_mean(1-tf.cos(gt_angle-pred_angle))


# geo map L1 loss
def lossGeo(labels,predictions):#,scope=tf.GraphKeys.LOSSES):
    # with tf.variable_scope(scope):
        batch_size,w,h,_,d = labels.shape
        print('dddddd',batch_size,w,h,_, d)
        batch_size_1,w_1,h_1,_,d_1 = labels.shape
        print(batch_size_1,w_1,h_1,_,d_1)
        diff = tf.abs(labels-predictions)
        # less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
        # smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)#同上图公式
        # # print('smooth_l1_loss',smooth_l1_loss.shape)
        smooth_l1_loss = diff
        smooth_l1_loss= tf.reduce_sum(smooth_l1_loss,axis=-2)# = tf.add(smooth_l1_loss,axis=-2)
        print(smooth_l1_loss.shape)
        return tf.math.log(tf.reduce_mean(smooth_l1_loss)) #取平均值 tf.reduce_mean(tf.reshape(smooth_l1_loss,(batch_size,w,h,3))))#


def Smooth_l1_loss(labels,predictions):#,scope=tf.GraphKeys.LOSSES):
    # with tf.variable_scope(scope):
	    diff=tf.abs(labels-predictions)
	    less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
	    smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)#同上图公式
	    return tf.reduce_mean(smooth_l1_loss) #取平均值


ops = keras.optimizers.Adam(2e-4,beta_1=0.5)


@tf.function
def train_step(train_data,label_score_img,label_abgle_img,label_geo_img):
    #persistent true 为了重复计算梯度
    with tf.GradientTape(persistent=True) as tape:

        pred_score,angle_map,geo_map = net(train_data,training=True)#

        loss_score = scoreMapLossDice(label_score_img,pred_score)
        # print(gt_loss)
        # print('vefore',)
        loss_angle =  0#lossAngle(label_abgle_img,angle_map)
        loss_geo =  Smooth_l1_loss(label_geo_img,geo_map)
        loss_all = loss_score +loss_geo# 10*loss_angle 


    gradient = tape.gradient(loss_all,net.trainable_variables)
    print(net.trainable_variables)

    ops.apply_gradients(zip(gradient,net.trainable_variables))

    print('4')
    return loss_score,loss_angle,loss_geo


def fit(train_data_zip ,epochs):
    i =0
    for epoch in range(epochs):
        print(epoch)
        for t,s,a,g in train_data_zip:#zip(train_data,label_score_img,label_abgle_img,label_geo_img):,a,g

            loss_score,loss_angle,loss_geo = train_step(t,s,a,g)
        #     if  loss_score == np.nan or  loss_angle==np.nan or loss_geo==np.nan or loss_score == np.inf or  loss_angle==np.inf or loss_geo==np.inf or loss_score == np.NINF or  loss_angle==np.NINF or loss_geo==np.NINF:
        #             break
#             print('loss_geo.shape is {}'.format(loss_geo.size))
            print("i={},loss_score is {},loss_angle is {},loss_geo is {}".format(i,loss_score,loss_angle,loss_geo))
            i = i+1
            if i > 0 and i%100 == 0:

                net.save('net.h5')
        # if  loss_score == np.nan or  loss_angle==np.nan or loss_geo==np.nan or loss_score == np.inf or  loss_angle==np.inf or loss_geo==np.inf or loss_score == np.NINF or  loss_angle==np.NINF or loss_geo==np.NINF:
        #                             break


fit(train_data_zip,100000)

net.save('net.h5')

