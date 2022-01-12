# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:26:58 2021

@author: laakom
"""


from __future__ import print_function
import tensorflow.keras as ks
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from models import resnet_v1 , resnet_v2,mobilenets,resnet_model 

import numpy as np
import os

import tensorflow as tf 
import imagenet_input

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Training parameters
batch_size =  256
epochs = 100
alpha = 0.0001  # weight decay coefficient
num_classes = 1000
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy


#specify the diroctory of the TF records of Imagenet.
dataset = 'imagenet'
dataset_dir = '/scratch/zhangh/imagenet_tdfs/tfrecords/'
base_model = 'resnet50_'
# Choose what attention_module to use from the following options: 
# None: standard approach without attention
#'se_block' or 'cbam_block': original SE and CBAM appraoches.
#'se_block_ours' or 'se_block_ours_05' or 'se_block_ours_08': First variant of our approach using SE attention with alpha=1, alpha=0.5, and alpha=0.8 respectively.  
#'cbam_block_ours' or 'cbam_block_ours_05' or 'cbam_block_ours_08': First variant of our approach using CBAM attention with alpha=1, alpha=0.5, and alpha=0.8 respectively.  
# 'se_block_ours_invsigmoid'  or 'cbam_block_ours_invsigmoid': Second variant of our approach using SE or CBAM approach.
# 'se_block_ours_msigmoid'  or 'cbam_block_ours_msigmoid': Third variant of our approach using SE or CBAM approach.

attention_module =None # 'se_block'  'se_block_ours' 'se_block_ours_05' 'se_block_ours_invsigmoid' 'se_block_ours_msigmoid' 'cbam_block_ours' 'cbam_block_ours_msigmoid'  'cbam_block_ours_invsigmoid'

model_type = base_model if attention_module==None else base_model+'_'+attention_module + '_'
model_name =  dataset + '_' + model_type  + '.hdf5'


USE_BFLOAT16 = False



def lr_schedule(epoch):
    lr = 0.1
    if epoch > 80:
        lr = 0.0001
    elif epoch > 60:
        lr = 0.001
    elif epoch > 30:
        lr = 0.01
    print('Learning rate: ', lr)
    return lr







imagenet_train = imagenet_input.ImageNetInput(
    is_training=True, data_dir=dataset_dir, batch_size=batch_size,
    use_bfloat16=USE_BFLOAT16)
imagenet_eval = imagenet_input.ImageNetInput(
    is_training=False, data_dir=dataset_dir, batch_size=batch_size,
    use_bfloat16=USE_BFLOAT16)






# Open a strategy scope.
with strategy.scope():
    input_shape = (224,224,3)
    model = resnet_model.ResNet50(num_classes=num_classes,attention_module=attention_module)

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0),momentum=0.9,nesterov=True), 
              metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])


model.summary()
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
if os.path.exists(filepath):
    print('warning loading existing module')
    model.load_weights(filepath)
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             save_best_only=False,
                             verbose=1,
                             save_weights_only=True)

lr_scheduler =  LearningRateScheduler(lr_schedule)               #lr_schedule_warmup(1281167 // batch_size)


savecsvlog = CSVLogger(filepath[:-len('.hdf5')] + '_log.csv', separator=',', append=True )    

callbacks = [checkpoint, lr_scheduler,savecsvlog]



model.fit(
   imagenet_train.input_fn(),
   callbacks=callbacks,
   steps_per_epoch=1281167 // batch_size,
   validation_data=imagenet_eval.input_fn(),
   validation_steps=50000 // batch_size,
   initial_epoch=0,
   workers=64,
   epochs=epochs)

#model.load_weights(filepath[:-len('.hdf5')] + '_final.hdf5')	
# Score trained model.
scores = model.evaluate(x = imagenet_eval.input_fn(), workers=64 ,  epochs=epochs ,steps=50000 // batch_size, verbose=1)
print('Model: ', model_name)
print('Final Test loss:', scores[0])
print('Final Test accuracy:', scores[1])
print('Final top-5 accuracy:', scores[2])

