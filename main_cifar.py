# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:38:46 2021

@author: laakom
"""

from __future__ import print_function
import tensorflow.keras as ks
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from models import resnet_v1 ,mobilenets,densenet  
import numpy as np
import os
import json

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Training parameters
batch_size =  128
epochs = 200
data_augmentation = True

#specify the dataset to use: CIFAR or CIFAR 100:
dataset = 'cifar10'  # or 'cifar100'
num_classes = 10 # or 100


base_model = 'Resnet50v1'  # 'densenet' # or   'mobileNet'

# Choose what attention_module to use from the following options: 
# None: standard approach without attention
#'se_block' or 'cbam_block': original SE and CBAM appraoches.
#'se_block_ours' or 'se_block_ours_05' or 'se_block_ours_08': First variant of our approach using SE attention with alpha=1, alpha=0.5, and alpha=0.8 respectively.  
#'cbam_block_ours' or 'cbam_block_ours_05' or 'cbam_block_ours_08': First variant of our approach using CBAM attention with alpha=1, alpha=0.5, and alpha=0.8 respectively.  
# 'se_block_ours_invsigmoid'  or 'cbam_block_ours_invsigmoid': Second variant of our approach using SE or CBAM approach.
# 'se_block_ours_msigmoid'  or 'cbam_block_ours_msigmoid': Third variant of our approach using SE or CBAM approach.

attention_module =None # 'se_block'  'se_block_ours' 'se_block_ours_05' 'se_block_ours_invsigmoid' 'se_block_ours_msigmoid' 'cbam_block_ours' 'cbam_block_ours_msigmoid'  'cbam_block_ours_invsigmoid'



model_type = base_model if attention_module==None else base_model+'_'+attention_module 

# number of runs
iterations = 3

conf_name =dataset + '_'  + base_model
full_path = conf_name+ '_' + 'vanilla' if attention_module==None else   conf_name+ '_' + attention_module    

json_name=  full_path+ '.json'

input_shape=(32,32,3) 


def lr_schedule(epoch):
    lr=0.1
    if epoch > 160:
      lr = 0.0008 
    elif epoch > 120:
      lr = 0.004 
    elif epoch > 60:
       lr =  0.02             
    return lr




def build_model(base_model='Resnet50v1',depth = 50,attention_module=None):    
    # create the base pre-trained model

     # For ResNet, specify the depth (e.g. ResNet50: depth=50)
    if base_model == 'Resnet50v1':
        depth = 50
        model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth,num_classes=num_classes, attention_module=attention_module)    
    if base_model == 'mobileNet':
        model =  mobilenets.MobileNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
    if base_model == 'densenet':
        model = densenet.DenseNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module,reduction=0.0,bottleneck=True) 
    
    model.summary()           
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0),momentum=0.9),
              metrics=['accuracy','top_k_categorical_accuracy'])
    return  model



def process_dataset(name= 'cifar10'):
    print('importing ' + name + '... \n' )
    if name == 'cifar10':
        num_classes = 10
        fashion_mnist = ks.datasets.cifar10   
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  
    if name == 'cifar100':
        num_classes = 100
        cfar_large = ks.datasets.cifar100   
        (train_images, train_labels), (test_images, test_labels) = cfar_large.load_data() 
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
   
    train_images = train_images / 255.0
    test_images = test_images / 255.0     
    train_labels = ks.utils.to_categorical(train_labels, num_classes)
    test_labels = ks.utils.to_categorical(test_labels, num_classes)
   
    return (train_images, train_labels), (test_images, test_labels)



def evaluate_model(train_images, train_labels,test_images, test_labels, base_model='Resnet50v1',attention_module= None):
    train_data =train_images[:40000]
    train_label= train_labels[:40000]
    val_data =train_images[40000:]
    val_label= train_labels[40000:]

    x_train_mean = np.mean(train_data, axis=0)
    train_data -= x_train_mean
    val_data -= x_train_mean
    test_images -= x_train_mean

    accuracy = np.zeros(iterations)
    top5accuracies= np.zeros(iterations)
    for i in range(iterations):

        my_model = build_model(base_model=base_model,attention_module=attention_module)

        # Prepare callbacks for model saving and for learning rate adjustment.
        model_name = 'model_iteration'  + str(i) + '.hdf5'
        
        filepath =  os.path.join(full_path, model_name) 
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=10,
                                       min_lr=0.5e-6)
        
        savecsvlog = CSVLogger(filepath[:-len('.hdf5')] + '_log.csv', separator=',', append=True )    
        
        callbacks = [checkpoint, lr_reducer, lr_scheduler,savecsvlog]
        
        datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.125,
                # randomly shift images vertically
                height_shift_range=0.125,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
                      
                      
     
        datagen.fit(train_data)                

        my_model.fit(datagen.flow(train_data, train_label, batch_size=batch_size), validation_data=(val_data, val_label),
                                   workers=32,callbacks=callbacks , epochs=epochs)
                
              
        my_model.load_weights(filepath)
   
            
       #compute scores
        scores =  my_model.evaluate(test_images, test_labels, verbose=0)
    
        accuracy[i] = 100 *scores[1]
        save_nametxt = os.path.join(full_path, 'best_accuracy.txt')
        np.savetxt(save_nametxt,accuracy)
 
        top5accuracies[i] = 100 * scores[2]
        save_nametxt = os.path.join(full_path, 'best_top5_accuracy.txt')
        np.savetxt(save_nametxt,top5accuracies)        
        
    mean_acc = np.mean(accuracy)
    std_acc =  np.std(accuracy)

    mean_top5acc = np.mean(top5accuracies)
    std_top5acc =  np.std(top5accuracies)    
    return mean_acc,std_acc,mean_top5acc,std_top5acc













try:
    os.mkdir(full_path)
except:
    print('directory exisits: ' + full_path)



(train_images, train_labels), (test_images, test_labels) =  process_dataset(name= dataset)



mean_acc,std_acc,mean_top5acc,std_top5acc =evaluate_model(train_images, train_labels,test_images, test_labels,base_model = base_model,attention_module=attention_module)

print('Test error for ' + full_path + ': ', 100 - mean_acc, ' std=',std_acc)

print('top 5 Test error for ' + full_path + ': ', 100 - mean_top5acc, ' std=',std_top5acc)

total_errors_statistics = [full_path , 100.0 - mean_acc, std_acc,100.0 - mean_top5acc, std_top5acc ]

json.dump( total_errors_statistics, open(json_name, 'w' ) )












