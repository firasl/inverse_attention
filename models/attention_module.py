from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
import tensorflow as tf

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
epsilon = 1e-6
def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'se_block_08': # SE_block
    net = se_block(net, alpha=0.8)
  elif attention_module == 'se_block_05': # SE_block
    net = se_block(net, alpha=0.5)


  elif attention_module == 'se_block_ours': # SE_block
    net = se_block_ours(net,variant='standard')
  elif attention_module == 'se_block_ours_05': # SE_block
    net = se_block_ours(net,variant='standard', alpha=0.5)
  elif attention_module == 'se_block_ours_08': # SE_block
    net = se_block_ours(net,variant='standard', alpha=0.8)

  elif attention_module == 'se_block_ours_invsigmoid': # SE_block
    net = se_block_ours(net,variant='invsigmoid')

  elif attention_module == 'se_block_ours_invsigmoidraw': # SE_block
    net = se_block_ours(net,variant='invsigmoidraw')


  elif attention_module == 'se_block_ours_msigmoid': # SE_block
    net = se_block_ours(net,variant='msigmoid')


  elif attention_module == 'se_block_ours_mexp': # SE_block
    net = se_block_ours(net,variant='mexp')
  elif attention_module == 'se_block_ours_mexpsigm': # SE_block
    net = se_block_ours(net,variant='mexpsigm')
    
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  elif attention_module == 'cbam_block_08': # CBAM_block
    net = cbam_block(net, alpha=0.8)
  elif attention_module == 'cbam_block_05': # CBAM_block
    net = cbam_block(net, alpha=0.5)
  elif attention_module == 'cbam_block_ours': # CBAM_block
    net = cbam_block_ours(net,variant='standard') 
  elif attention_module == 'cbam_block_ours_08': # CBAM_block
    net = cbam_block_ours(net,variant='standard', alpha=0.8)  
  elif attention_module == 'cbam_block_ours_05': # CBAM_block
    net = cbam_block_ours(net,variant='standard', alpha=0.5)  
  elif attention_module == 'cbam_block_ours_msigmoid': # SE_block
    net = cbam_block_ours(net,variant='msigmoid')

  elif attention_module == 'cbam_block_ours_invsigmoid': # SE_block
    net = cbam_block_ours(net,variant='invsigmoid')
  elif attention_module == 'cbam_block_ours_mexp': # SE_block
    net = cbam_block_ours(net,variant='mexp')
  elif attention_module == 'cbam_block_ours_mexpsigm': # SE_block
    net = cbam_block_ours(net,variant='mexpsigm')
        
    
    
    
    
    
    
    
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8, alpha=1.0):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)
	se_feature = alpha*se_feature
	se_feature = multiply([input_feature, se_feature])
	return se_feature



def se_block_ours(input_feature, ratio=8,variant='standard', alpha=1.0):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	if variant=='standard':
		se_feature = Dense(channel,
    					   activation='sigmoid',
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
		se_feature = 1.0- alpha*se_feature
	elif variant=='invsigmoid':
		se_feature = Dense(channel,
    					   activation='sigmoid',
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
            
		se_feature = 1/ (se_feature +epsilon)
		se_feature = Activation('sigmoid')(se_feature)

	elif variant=='invsigmoidraw':
		se_feature = Dense(channel,
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
            
		se_feature = 1/ (se_feature +epsilon)
		se_feature = Activation('sigmoid')(se_feature)

	elif variant=='mexp':
		se_feature = Dense(channel,
    					   activation='relu',
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
            
		se_feature = tf.math.exp(-1.0*se_feature)
		#se_feature = Activation('sigmoid')(se_feature)
	elif variant=='mexpsigm':
		se_feature = Dense(channel,
    					   activation='relu',
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
            
		se_feature = tf.math.exp(-1.0*se_feature)
		se_feature = Activation('sigmoid')(se_feature)

	elif variant=='msigmoid':
		se_feature = Dense(channel,
    					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
    					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
    					   use_bias=True,
    					   bias_initializer='zeros')(se_feature)
		assert se_feature.shape[1:] == (1,1,channel)
		if K.image_data_format() == 'channels_first':
			se_feature = Permute((3, 1, 2))(se_feature)
            
		se_feature = -1.0 * se_feature
		se_feature = Activation('sigmoid')(se_feature)

    
	se_feature = multiply([input_feature, se_feature])
	return se_feature








def cbam_block(cbam_feature, ratio=8, alpha=1.0):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio, alpha=alpha)
	cbam_feature = spatial_attention(cbam_feature, alpha=alpha)
	return cbam_feature


def cbam_block_ours(cbam_feature, ratio=8,variant='standard', alpha=1.0):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention_ours(cbam_feature, ratio,variant=variant, alpha=alpha)
	cbam_feature = spatial_attention_ours(cbam_feature,variant=variant, alpha=alpha)
	return cbam_feature


def channel_attention(input_feature, ratio=8, alpha=1.0):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1

	channel = input_feature.shape[channel_axis]    
#   	channel = tf.shape(input_feature)[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   use_bias=True,
					   bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	cbam_feature = alpha*cbam_feature
	return multiply([input_feature, cbam_feature])


def channel_attention_ours(input_feature, ratio=8,variant='standard', alpha=1.0):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1

	channel = input_feature.shape[channel_axis]    
#   	channel = tf.shape(input_feature)[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
					   kernel_initializer=initializers.RandomNormal(stddev=0.01),
					   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					   bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])

	if variant=='standard':	  
		cbam_feature = Activation('sigmoid')(cbam_feature)	
		cbam_feature = 1 - alpha*cbam_feature
	elif variant=='invsigmoid':	  
		cbam_feature = Activation('sigmoid')(cbam_feature)	
		cbam_feature = 1/ (cbam_feature +epsilon)
		cbam_feature = Activation('sigmoid')(cbam_feature)

	elif variant=='mexp':	  
		cbam_feature = Activation('relu')(cbam_feature)	

		cbam_feature = tf.math.exp(-1.0*cbam_feature)

	elif variant=='mexpsigm':	   
			  
		cbam_feature = Activation('relu')(cbam_feature)	

		cbam_feature = tf.math.exp(-1.0*cbam_feature)

		cbam_feature = Activation('sigmoid')(cbam_feature)

	elif variant=='msigmoid':	  
			
		cbam_feature = -1.0* cbam_feature
		cbam_feature = Activation('sigmoid')(cbam_feature)


    
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature]) 












def spatial_attention(input_feature, alpha=1.0):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	cbam_feature = alpha*cbam_feature
	return multiply([input_feature, cbam_feature])
		


def spatial_attention_ours(input_feature, variant='standard', alpha=1.0):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	if variant=='standard':	
		cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
		assert cbam_feature.shape[-1] == 1
		cbam_feature = 1 - alpha*cbam_feature

	elif variant=='invsigmoid':
		cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
		assert cbam_feature.shape[-1] == 1
		cbam_feature = 1/ (cbam_feature +epsilon)
		cbam_feature = Activation('sigmoid')(cbam_feature)
        
	elif variant=='mexp':	   
		cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
		assert cbam_feature.shape[-1] == 1
		cbam_feature = tf.math.exp(-1.0*cbam_feature)
	elif variant=='mexpsigm':	   
		cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
		assert cbam_feature.shape[-1] == 1
		cbam_feature = tf.math.exp(-1.0*cbam_feature)
		cbam_feature = Activation('sigmoid')(cbam_feature)

	elif variant=='msigmoid':	   
		cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
					use_bias=False)(concat)	
		assert cbam_feature.shape[-1] == 1
		cbam_feature = -1.0*cbam_feature
		cbam_feature = Activation('sigmoid')(cbam_feature)


	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])
		

