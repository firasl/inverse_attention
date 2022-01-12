# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.
Adapted from tf.keras.applications.resnet50.ResNet50().
Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from models.attention_module import attach_attention_module
from tensorflow.python.keras.engine import training

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def identity_block(input_tensor, kernel_size, filters, stage, block,attention_module=None):
  """The identity block is the block that has no conv layer at shortcut.
  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, kernel_size, use_bias=False,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

# ....
  if attention_module is not None:
                x = attach_attention_module(x, attention_module)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               attention_module=None):
  """A block that has a conv layer at shortcut.
  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
  Returns:
      Output tensor for the block.
  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                    use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

  shortcut = layers.Conv2D(filters3, (1, 1), use_bias=False, strides=strides,
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                           name=conv_name_base + '1')(input_tensor)
  shortcut = layers.BatchNormalization(axis=bn_axis,
                                       momentum=BATCH_NORM_DECAY,
                                       epsilon=BATCH_NORM_EPSILON,
                                       name=bn_name_base + '1')(shortcut)

  if attention_module is not None:
                x = attach_attention_module(x, attention_module)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def ResNet50(num_classes,attention_module=None):
  """Instantiates the ResNet50 architecture.
  Args:
    num_classes: `int` number of classes for image classification.
  Returns:
      A Keras model instance.
  """
  # Determine proper input shape
  if backend.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
    bn_axis = 1
  else:
    input_shape = (224, 224, 3)
    bn_axis = 3

  img_input = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = layers.Conv2D(64, (7, 7), use_bias=False,
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name='conv1')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name='bn_conv1')(x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),attention_module=attention_module)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',attention_module=attention_module)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',attention_module=attention_module)

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',attention_module=attention_module)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',attention_module=attention_module)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',attention_module=attention_module)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',attention_module=attention_module)

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',attention_module=attention_module)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',attention_module=attention_module)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',attention_module=attention_module)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',attention_module=attention_module)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',attention_module=attention_module)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',attention_module=attention_module)

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',attention_module=attention_module)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',attention_module=attention_module)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',attention_module=attention_module)

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = layers.Dense(
      num_classes,
      activation='softmax',
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      name='fc1000')(
          x)

  # Create model.
  return models.Model(img_input, x, name='resnet50')













def identity_block18(input_tensor, kernel_size, filters, stage, block,attention_module=None):
  """The identity block is the block that has no conv layer at shortcut.
  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
  Returns:
      Output tensor for the block.
  """
  filters1, filters2 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

# ....
  if attention_module is not None:
                x = attach_attention_module(x, attention_module)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block18(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               attention_module=None):
  """A block that has a conv layer at shortcut.
  Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
  Returns:
      Output tensor for the block.
  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, (1, 1), use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

  shortcut = layers.Conv2D(filters2, (1, 1), use_bias=False, strides=strides,
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                           name=conv_name_base + '1')(input_tensor)
  shortcut = layers.BatchNormalization(axis=bn_axis,
                                       momentum=BATCH_NORM_DECAY,
                                       epsilon=BATCH_NORM_EPSILON,
                                       name=bn_name_base + '1')(shortcut)

  if attention_module is not None:
                x = attach_attention_module(x, attention_module)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x





def ResNet18_like50(num_classes,attention_module=None):
  """Instantiates the ResNet50 architecture.
  Args:
    num_classes: `int` number of classes for image classification.
  Returns:
      A Keras model instance.
  """
  # Determine proper input shape
  if backend.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
    bn_axis = 1
  else:
    input_shape = (224, 224, 3)
    bn_axis = 3

  img_input = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = layers.Conv2D(64, (7, 7), use_bias=False,
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name='conv1')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name='bn_conv1')(x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
  x = identity_block18(x, 3, [64, 64], stage=2, block='a', attention_module=attention_module)
  x = identity_block18(x, 3, [64, 64], stage=2, block='b',attention_module=attention_module)

  x = conv_block18(x, 3, [128, 128], stage=3, block='a',attention_module=attention_module)
  x = identity_block18(x, 3, [128, 128], stage=3, block='b',attention_module=attention_module)

  x = conv_block18(x, 3, [256, 256], stage=4, block='a',attention_module=attention_module)
  x = identity_block18(x, 3, [256, 256], stage=4, block='b',attention_module=attention_module)

  x = conv_block18(x, 3, [512, 512], stage=5, block='a',attention_module=attention_module)
  x = identity_block18(x, 3, [512, 512], stage=5, block='b',attention_module=attention_module)

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = layers.Dense(
      num_classes,
      activation='softmax',
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      name='fc1000')(
          x)

  # Create model.
  return models.Model(img_input, x, name='resnet18')




def block3(x,
          filters,
          kernel_size=3,
          stride=1,
          conv_shortcut=True,
          name=None, attention_module = None):
  """A residual block.
  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  if conv_shortcut:
    shortcut = layers.Conv2D(
        filters,
        1,
        strides=stride,
        use_bias=False,
        name=name + '_0_conv',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_1_pad')(x)
  x = layers.Conv2D(
      filters, kernel_size=kernel_size, strides=stride, use_bias=False, name=name + '_1_conv',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      filters, kernel_size=kernel_size, use_bias=False, name=name + '_2_conv',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)


  if attention_module is not None:
                x = attach_attention_module(x, attention_module)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x

def stack3(x, filters, blocks, stride1=2, conv_shortcut=True, name=None, attention_module = None):
  """A set of stacked residual blocks.
  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    groups: default 1, group size for grouped convolution.
    base_width: default 64, filters of the intermediate layer in a block.
    name: string, stack label.
  Returns:
    Output tensor for the stacked blocks.
  """
  x = block3(x, filters, stride=stride1, conv_shortcut=conv_shortcut, name=name + '_block1', attention_module = attention_module)
  for i in range(2, blocks + 1):
    x = block3(
        x,
        filters,
        conv_shortcut=False,
        name=name + '_block' + str(i), attention_module = attention_module)
  return x

def ResNet18(include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000, attention_module=None,
            **kwargs):
  """Instantiates the ResNet18 architecture."""
  def stack_fn(x):
    x = stack3(x, 64, 2, stride1=1, conv_shortcut=False, name='conv2', attention_module = attention_module)
    x = stack3(x, 128, 2, name='conv3', attention_module = attention_module)
    x = stack3(x, 256, 2, name='conv4', attention_module = attention_module)
    return stack3(x, 512, 2, name='conv5', attention_module = attention_module)
  return ResNet(stack_fn, False, 'resnet18', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)







def ResNet(stack_fn,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           deep_stem=False,
           stem_width=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
  """Instantiates the ResNeSt, ResNeXt, and Wide ResNet architecture.
  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.resnet.preprocess_input` for an example.
  Arguments:
    stack_fn: a function that returns output tensor for the
      stacked residual blocks.
    use_bias: whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
    model_name: string, model name.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.
  Returns:
    A `keras.Model` instance.
  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if 'layers' in kwargs:
    global layers
    layers = kwargs.pop('layers')
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', 'ssl', 'swsl', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), `ssl` '
                     '(semi-supervised), `swsl` '
                     '(semi-weakly supervised), '
                     'or the path to the weights file to be loaded.')
  if (weights == 'imagenet' or weights == 'ssl' or weights == 'swsl') and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"`, '
                     'or `weights` as `"ssl"`, '
                     'or `weights` as `"swsl"`, '
                     'with `include_top` '
                     ' as true, `classes` should be 1000')
  # Determine proper input shape
  input_shape = (224,224,3) 
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  if deep_stem:
    # Deep stem based off of ResNet-D
    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_0_pad')(img_input)
    x = layers.Conv2D(stem_width, 3, strides=2, use_bias=use_bias, name='conv1_0_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_0_bn')(x)
    x = layers.Activation('relu', name='conv1_0_relu')(x)
    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_1_pad')(x)
    x = layers.Conv2D(stem_width, 3, strides=1, use_bias=use_bias, name='conv1_1_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_1_bn')(x)
    x = layers.Activation('relu', name='conv1_1_relu')(x)
    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_2_pad')(x)
    x = layers.Conv2D(stem_width * 2, 3, strides=1, use_bias=use_bias, name='conv1_2_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_2_bn')(x)
    x = layers.Activation('relu', name='conv1_2_relu')(x)
  else:
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
  x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
  x = stack_fn(x)
  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)
  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name=model_name)
  return model
