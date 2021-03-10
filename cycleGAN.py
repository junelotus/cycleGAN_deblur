#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
# get_ipython().run_line_magic('matplotlib', 'inline')
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math


# In[2]:


tf.__version__


# In[3]:


os.listdir('./')


# In[4]:


img_ori =glob.glob('dataset/sharp_to_split/*jpg') #all apple image to convert to original


# In[5]:


len(img_ori)


# In[6]:


img_target = glob.glob('dataset/blur_second_to_split/*jpg') # all orange images


# In[7]:


len(img_target)


# In[ ]:




# place test image
# In[8]:


# 图此读取&解码函数
def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img
    


# In[9]:


def load_img(path):
    img = read_jpg(path)
    img = tf.image.resize(img,(256,256)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


# In[10]:


# 创建dataset
train_a = tf.data.Dataset.from_tensor_slices(img_ori)
train_b = tf.data.Dataset.from_tensor_slices(img_target)


# In[ ]:





# In[11]:


buffer_size  = 20


# In[12]:


#dataset 按照load_img 来加载
train_a = train_a.map(load_img,num_parallel_calls = AUTOTUNE).cache().shuffle(buffer_size).batch(1)

train_b= train_b.map(load_img,num_parallel_calls = AUTOTUNE).cache().shuffle(buffer_size).batch(1)
#cache的作用是缓存，缓存可以大大提高读取的效率 在cache里面添加路径参数，可以将下一批将要读取的数据缓存到此路径中
#batch 为1 是因为不同的图片可以是局部或者全部处理，每一张图片都有自己的转换风格的方式，所以要一张一张来处理


# In[13]:


# zip函数 创建了数据 有两部分数据 
data_train = tf.data.Dataset.zip((train_a,train_b)) 


# In[14]:


data_train


# In[15]:


for apple,orange in data_train.take(1):
       plt.subplot(1,2,1) # 三个参数分别表示：表示画 1行2 列 中的第1个
       plt.imshow(keras.preprocessing.image.array_to_img(apple[0])) # 从array 转换为img
       plt.subplot(1,2,2)
       plt.imshow(keras.preprocessing.image.array_to_img(orange[0])) # 从array 转换为img


# In[16]:


# 类似于batchNornalization 的 Instance normalization，后者非常适合style transfer，直接对一张图片进行归一化；
# BN适用于判别器，比如对图片进行分类；因为BN注重于对每个batch进行归一化，从而保证数据分布的一致性，
#而判别器模型的结果正式取决于数据的整体分布，我们希望此批次的分布和整体数据额分布是一致的

#BN对batch的size大小比较敏感，由于每次计算的 均值和方差是在一个batch上，若batchsize太小，则计算的均值和方差不足以代表整体的数据分布

#Batch size 在内存显存允许的情况下 越大越好


# In[17]:


# instance  noramlization 封装爱tensorflow_addons
# bacth(1)的原因
#IN 适用于生成器，比如图片的风格迁移，因为每张图片的风格不同，对整个batch进行归一化不适合图像风格化，使用IN不仅可以加速模型手链，
#并且可以保持每张图像实例之间的独立性

#实际上是对单张图片 沿着channel进行均值和方差的计算


# In[18]:


#!pip --default-timeout=1000 install -i https://pypi.tuna.tsinghua.edu.cn/simple  tensorflow_addons
import tensorflow_addons as tfa


# In[19]:


def downsample(filters,size, apply_batchnorm=True):# 卷积核个数和大小
    result = keras.Sequential()#顺序模型的创建
    result.add(keras.layers.Conv2D(filters,size,strides = 2,padding='same',use_bias=False)
              )
    if apply_batchnorm:
        result.add(tfa.layers.InstanceNormalization())
        result.add(keras.layers.LeakyReLU())
    return result


# In[20]:


def upsample(filters,size,drop=False):# 卷积核个数和大小,上采样过程中 为了过拟合 添加drop
    result = keras.Sequential()#顺序模型的创建
    result.add(keras.layers.Conv2DTranspose(filters,size,strides = 2,padding='same',
                                        use_bias=False))
    result.add(tfa.layers.InstanceNormalization())
    if drop:
         result.add(keras.layers.Dropout(0.5))   
    result.add(keras.layers.ReLU())
    return result


# In[21]:


lists_esc = []
lists_ebc = []
lists = []


# In[22]:


# 生成器使用改进的u-net模型的变形
def encoder():
    inputs = keras.layers.Input(shape=[256,256,3])
    down_stack=[
        downsample(64,4, apply_batchnorm=False),#128
        downsample(128,4),#64
        downsample(256,4),#32
        downsample(512,4),#16
        
        downsample(512,4),#8
        downsample(512,4),#4
        downsample(512,4),#2
        downsample(512,4),#1
    ]
    
    up_stack=[
        upsample(512,4,drop=True),#2
        upsample(512,4,drop=True),#4
        upsample(512,4,drop=True),#8
        upsample(512,4),#16
        
        upsample(256,4),#32
        upsample(128,4),#64
        upsample(64,4),#128
    ]
    last = keras.layers.Conv2DTranspose(3,4,strides = 2,padding ='same',activation='tanh')# 输出层使用tanh激活到【-1,1】 每个像素都在【-1,1之间】
    lists = []
    x = inputs
    for down in down_stack:
        x = down(x)
        lists.append(x)
       
    lists = reversed(lists  [:-1])# 不要最后一层 并且翻转
    for up,skip in zip(up_stack,lists):
        x = up(x)
        x = keras.layers.Concatenate()([x,skip]) # 通道数增加一倍
        
    x = last(x)
    # 创建模型并且输出
    return keras.Model(inputs = inputs,outputs = x)
    
    


# In[23]:


e =encoder()
# lists


# In[24]:


# 生成器使用改进的u-net模型的变形
def generator():
    inputs = keras.layers.Input(shape=[256,256,6])
    down_stack=[
        downsample(64,4, apply_batchnorm=False),#128
        downsample(128,4),#64
        downsample(256,4),#32
        downsample(512,4),#16
        
        downsample(512,4),#8
        downsample(512,4),#4
        downsample(512,4),#2
        downsample(512,4),#1
    ]
    
    up_stack=[
        upsample(512,4,drop=True),#2
        upsample(512,4,drop=True),#4
        upsample(512,4,drop=True),#8
        upsample(512,4),#16
        
        upsample(256,4),#32
        upsample(128,4),#64
        upsample(64,4),#128
    ]
    last = keras.layers.Conv2DTranspose(3,4,strides = 2,padding ='same',activation='tanh')# 输出层使用tanh激活到【-1,1】 每个像素都在【-1,1之间】
    lists = []
    x = inputs
    for down in down_stack:
        x = down(x)
        lists.append(x)
       
    lists = reversed(lists  [:-1])# 不要最后一层 并且翻转
    for up,skip in zip(up_stack,lists):
        x = up(x)
        x = keras.layers.Concatenate()([x,skip]) # 通道数增加一倍
        
    x = last(x)
    # 创建模型并且输出
    return keras.Model(inputs = inputs,outputs = x)
    
    


# In[25]:


# lists_esc = esc()
gen = generator()


# In[ ]:





# In[ ]:





# In[26]:


def discriminator():
    inputs  = keras.layers.Input(shape=[256,256,3])
    down1 = downsample(64,4,apply_batchnorm=False)(inputs) # apply_batchnorm
    down2 = downsample(128,4)(down1)
    down3 = downsample(256,4)(down2)#32
    # patch gan 很适合对图像的一部分进行判别
    # 首先填充
    zero_pad1 = keras.layers.ZeroPadding2D()(down3)  # 上下左右各添加一行、列0，变成34*34*256
    
    conv = keras.layers.Conv2D(512,4,strides = 1,use_bias=False)(zero_pad1)  # 31*31*512 ,padding为valid
    norm = tfa.layers.InstanceNormalization()(conv)
    leakrelu = keras.layers.LeakyReLU()(norm)
    
    zero_pad2 = keras.layers.ZeroPadding2D()(leakrelu)  # 上下左右各添加一行、列0，变成33*33*256
    
    last = keras.layers.Conv2D(1,4,strides = 1)(zero_pad2)  # 30*30*512 ,padding为valid ,即为 33-（4-1）
    
    return keras.Model(inputs = inputs,outputs = last)


# In[27]:


disc =discriminator()


# In[28]:


# 损失函数 和 优化函数
# 判别器损失 生成器损失 循环一致损失，生成的橘子的苹果图像反回去的时候 和原始的苹果图像一致的差距

loss_obj =keras.losses.BinaryCrossentropy(from_logits=True) # 二元交叉熵损失
# kld_loss = keras.losses.kld()


# In[29]:


def dis_loss(real_image,gen_image):
    real_loss = loss_obj(tf.ones_like(real_image),real_image)
    gen_loss = loss_obj(tf.zeros_like(gen_image),gen_image)
    return real_loss+gen_loss


# In[30]:


def gen_loss(gen_image):
    gen_loss = loss_obj(tf.ones_like(gen_image),gen_image)# 希望可以骗过生成器,是对生成器的输出做的判定
    #循环一致损失
    return gen_loss


# In[31]:


def cycle_loss(real_img,cycled_img):# 是对图像做的判定
    return tf.reduce_mean(tf.abs(real_img-cycled_img))*10 # 也可以使用二阶（tf.square）损失，还有percistance loss 计算高阶特征时间的损失


# In[32]:


def kl_loss(blur):
#     shape = blur.shape
#     test = blur
#     i=1
#     while i<shape[0]:
#         test[i] =test[i]+test[i-1] 
#         i= i+1
#     i= 0    
#     mean = test/ shape[0]  
#     vae = 0
#     while i<shape[0]:
#         var =var +(blur[i]-mean)*(blur[i]-mean)
#         i = i+1
        
#     var = var/shape[0]
#     loss = 0
#     for i in range(3):
#         loss  = loss+mean*mean+var*var+math.log(var*var)-1
    
    mean,var = tf.nn.moments(x=blur, axes=[1,2])
    loss  = mean*mean+var*var+tf.math.log(var*var)
#     loss  = mean*mean+var*var+math.log(var*var)
    return 0.01*(loss/2.0)


# In[33]:


def L2_loss(sb,b):
    return 0.1*tf.reduce_mean(tf.square(sb-b))
    


# In[34]:


# 两个生成器 和 两个判别器
#优化器
gen_x_ops = keras.optimizers.Adam(2e-4,beta_1=0.5)
gen_y_ops = keras.optimizers.Adam(2e-4,beta_1=0.5)
disc_x_ops = keras.optimizers.Adam(2e-4,beta_1=0.5)
disc_y_ops = keras.optimizers.Adam(2e-4,beta_1=0.5)
eb_encoder_ops = keras.optimizers.Adam(2e-4,beta_1=0.5)
esc_encoder_opt =keras.optimizers.Adam(2e-4,beta_1=0.5)


# In[35]:


def show_image(model,test_input,aa):
    prediction = model(test_input,training=True)
    plt.figure(figsize =(15,15))
    display_list = [aa[0],prediction[0]]
    titles = ['input','predicted']
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i]*0.5+0.5)
        plt.axis('off')
    plt.show()    
    


# In[36]:


# gen_x : appleA to orangeA
# gen_y : orangeA to appleA'
# disc_x : 是否是真实的apple 主要针对生成的appleA‘
# disc_y:  是否是真实的orange，主要是针对生成的orangeA
esc_encoder = encoder()#  tf.keras.models.load_model('esc_encoder_wokl_2.h5')#encoder sharp content#
ebc_encoder =   encoder()#tf.keras.models.load_model('ebc_encoder_wokl_2.h5')# encoder()#encoder()#  encode blur content
eb_encoder =    encoder()#tf.keras.models.load_model('eb_encoder_wokl_2.h5')#encoder()# encoder()#encode blur
gen_b = generator()#tf.keras.models.load_model('gen_b_wokl_2.h5')#
# gen_s.save('gen_s.h5')
#                 gen_b.save('gen_b.h5')load_
gen_s =  generator()#tf.keras.models.load_model('gen_s_wokl_2.h5')#generator()#generator()#
disc_b = discriminator()# tf.keras.models.load_model('disc_b_wokl_2.h5')#
disc_s = discriminator()# tf.keras.models.load_model('disc_s_wokl_2.h5')#discriminator()#discriminator()#
    
    


# In[37]:


# keras.utils.plot_model(esc_encoder,'esc_encoder.png')
# keras.utils.plot_model(ebc_encoder,'ebc_encoder.png')
# keras.utils.plot_model(eb_encoder,'eb_encoder.png')


# In[37]:


#因为是一个循环的调用 密集的运算，可以定义为一个图运算
@tf.function
def train_step(image_s,image_b):
    #persistent true 为了重复计算梯度
    with tf.GradientTape(persistent=True) as tape:
        
        print('1')
        esc = esc_encoder(image_s,training=True) # 清晰男人内容
        ebc = ebc_encoder(image_b,training=True)# 模糊女人内容
        eb = eb_encoder(image_b,training=True)# 女人的模糊
        gen_s_input = tf.keras.layers.concatenate([ebc, eb])
        # 模糊女人内容+女人模糊 -》gen_s，生成的清晰女人
        gen_b_input = tf.keras.layers.concatenate([esc, eb])
        # 清晰男人内容+女人模糊 -> genb，生成的模糊男人
        
        
        fake_s = gen_s(gen_s_input, training=True) #清晰女人
        fake_b = gen_b(gen_b_input, training=True)# 模糊男人
        
        esc_fake = esc_encoder(fake_s,training=True)#清晰女人
        ebc_fake = ebc_encoder(fake_b,training=True)#模糊男人
        eb_fake = eb_encoder(fake_b,training=True)# 男人的模糊
        gen_s_input_fake = tf.keras.layers.concatenate([ebc_fake, eb_fake]) 
        #模糊男人 +男人的模糊 -》gen_s
        gen_b_input_fake = tf.keras.layers.concatenate([esc_fake, eb_fake])
        # 清晰女人+男人的模糊 -》gen_b
        
        print("gen_s_input_fake")
        print(gen_s_input_fake)
        
        cycle_s = gen_s(gen_s_input_fake, training=True)# 清晰男人

        cycle_b = gen_b(gen_b_input_fake, training=True)# 模糊女人
        
        disc_real_s  = disc_s(image_s, training=True)
        disc_fake_s = disc_s(fake_s, training=True)
        
        disc_real_b = disc_b(image_b, training=True)
        disc_fake_b = disc_b(fake_b, training=True)
        
        disc_loss_s  =  dis_loss(disc_real_s,disc_fake_s) 
        # z主要判别真假图片a 两张图片之间没有输入输出的因果关系
        disc_loss_b = dis_loss(disc_real_b,disc_fake_b)
        
#         KL_loss = kl_loss(eb)#+kl_loss(eb_fake)
        # 去模糊的图像和原始图像之间的
        ls_loss_ = L2_loss(image_b,fake_s) 
#         ls_loss_1 = L2_loss(esc,fake_b)
    
        
#         disc_loss_x_fake =  dis_loss(image_a,disc_real_a)
#         disc_loss_y_fake = dis_loss(image_b,disc_real_b)
        
        gen_loss_x = gen_loss(disc_fake_s)
        gen_loss_y = gen_loss(disc_fake_b)
        
        
        print('2')
        cycle_loss_ = cycle_loss(image_s,cycle_s) + cycle_loss(image_b,cycle_b) 
        
        total_gen_x_loss = gen_loss_x+cycle_loss_+ls_loss_
        total_gen_y_loss = gen_loss_y+cycle_loss_#+ls_loss_1
    # print('3')    
    # print(esc_encoder.trainable_variables)
#     esc_encoder_gradient = tape.gradient(ls_loss_1,esc_encoder.trainable_variables)
#     eb_encoder_gradient = tape.gradient(KL_loss,eb_encoder.trainable_variables)
    gen_x_gradient = tape.gradient(total_gen_x_loss,gen_s.trainable_variables)
    gen_y_gradient = tape.gradient(total_gen_y_loss,gen_b.trainable_variables)
    disc_x_gradient = tape.gradient(disc_loss_s,disc_s.trainable_variables)
    disc_y_gradient = tape.gradient(disc_loss_b,disc_b.trainable_variables)
    
    # 根据梯度 优化参数

#     esc_encoder_opt.apply_gradients(zip(esc_encoder_gradient,esc_encoder.trainable_variables))
#     eb_encoder_ops.apply_gradients(zip(eb_encoder_gradient,eb_encoder.trainable_variables))
    gen_x_ops.apply_gradients(zip(gen_x_gradient,gen_s.trainable_variables))
    gen_y_ops.apply_gradients(zip(gen_y_gradient,gen_b.trainable_variables))
    disc_x_ops.apply_gradients(zip(disc_x_gradient,disc_s.trainable_variables))
    disc_y_ops.apply_gradients(zip(disc_y_gradient,disc_b.trainable_variables))
    print('4')    
        
        
    


# In[38]:


# #因为是一个循环的调用 密集的运算，可以定义为一个图运算
# @tf.function
# def train_step(image_aa,image_bb):
#     #persistent true 为了重复计算梯度
#     with tf.GradientTape(persistent=True) as tape:
        
#         print('1')
#         image_a = esc_encoder(image_aa)#encoder sharp content
#         image_b = ebc_encoder(image_bb)# encode blur content
#         eb_encoder = encoder()# encode blur
#         fake_b = gen_x(image_a, training=True)
        
#         cycle_a = gen_y(fake_b, training=True)
        
#         fake_a = gen_y(image_b, training=True)
        
#         cycle_b = gen_x(fake_a, training=True)
        
#         disc_real_a  = disc_x(image_a, training=True)
#         disc_fake_a = disc_x(fake_a, training=True)
        
#         disc_real_b = disc_y(image_b, training=True)
#         disc_fake_b = disc_y(fake_b, training=True)
        
#         disc_loss_x  =  dis_loss(disc_real_a,disc_fake_a) # z主要判别真假图片a 两张图片之间没有输入输出的因果关系
#         disc_loss_y = dis_loss(disc_real_b,disc_fake_b)
        
# #         disc_loss_x_fake =  dis_loss(image_a,disc_real_a)
# #         disc_loss_y_fake = dis_loss(image_b,disc_real_b)
        
#         gen_loss_x = gen_loss(disc_fake_b)
#         gen_loss_y = gen_loss(disc_fake_a)
        
        
#         print('2')
#         cycle_loss_ = cycle_loss(image_a,cycle_a) + cycle_loss(image_b,cycle_b) 
        
#         total_gen_x_loss = gen_loss_x+cycle_loss_
#         total_gen_y_loss = gen_loss_y+cycle_loss_
#     print('3')    
#     gen_x_gradient = tape.gradient(total_gen_x_loss,gen_x.trainable_variables)
#     gen_y_gradient = tape.gradient(total_gen_y_loss,gen_y.trainable_variables)
#     disc_x_gradient = tape.gradient(disc_loss_x,disc_x.trainable_variables)
#     disc_y_gradient = tape.gradient(disc_loss_y,disc_y.trainable_variables)
    
#     # 根据梯度 优化参数
#     gen_x_ops.apply_gradients(zip(gen_x_gradient,gen_x.trainable_variables))
#     gen_y_ops.apply_gradients(zip(gen_y_gradient,gen_y.trainable_variables))
#     disc_x_ops.apply_gradients(zip(disc_x_gradient,disc_x.trainable_variables))
#     disc_y_ops.apply_gradients(zip(disc_y_gradient,disc_y.trainable_variables))
#     print('4')    
        
        
    


# In[39]:


epochs = 300


# In[40]:


def fit(train_ds,epochs):
    i = 0;
    for epoch in range(epochs):
        print("epoch={}".format(epoch))
        for a,b in train_ds:
            train_step(a,b)
            print("i={}".format(i))
            if i%500 == 0:
                print('.',end='')
                for aa,bb in train_ds.take(1):
                    esc = esc_encoder(aa,training=True) # 清晰男人内容
                    ebc = ebc_encoder(bb,training=True)# 模糊女人内容
                    eb = eb_encoder(bb,training=True)# 女人的模糊
                    gen_s_input = tf.keras.layers.concatenate([ebc, eb]) # 模糊女人+女人模糊 -》gen_s
                    gen_b_input = tf.keras.layers.concatenate([esc, eb]) # 清晰男人+女人模糊 -> genb
        
        
#                     fake_s = gen_s(gen_s_input, training=True) #清晰女人
#                     fake_b = gen_b(gen_b_input, training=True)# 模糊男人
                    
#                     show_image(gen_b,gen_b_input,aa)
#                     show_image(gen_s,gen_s_input,bb)
            i+=1
        if epoch%1==0 and epoch>1:
            for aa,bb in train_ds.take(1):
                esc = esc_encoder(aa,training=True) # 清晰男人内容
                ebc = ebc_encoder(bb,training=True)# 模糊女人内容
                eb = eb_encoder(bb,training=True)# 女人的模糊
                gen_s_input = tf.keras.layers.concatenate([ebc, eb]) # 模糊女人+女人模糊 -》gen_s
                gen_b_input = tf.keras.layers.concatenate([esc, eb]) # 清晰男人+女人模糊 -> genb
                show_image(gen_b,gen_b_input,aa)
                show_image(gen_s,gen_s_input,bb)
        
        
#                     fake_s = gen_s(gen_s_input, training=True) #清晰女人
#                     fake_b = gen_b(gen_b_input, training=True)# 模糊男人
#                 show_image(gen_s,gen_s_input,bb)
        if epoch%1 ==0 and epoch >=1:
                
                gen_s.save('gen_s_wokl_2.h5')
                gen_b.save('gen_b_wokl_2.h5')
                esc_encoder.save('esc_encoder_wokl_2.h5') 
                ebc_encoder.save('ebc_encoder_wokl_2.h5') 
                ebc_encoder.save('eb_encoder_wokl_2.h5') 
                disc_b.save('disc_b_wokl_2.h5') 
                disc_s.save('disc_s_wokl_2.h5') 
                
                
        


# In[ ]:


fit(data_train,epochs)







