
from keras import Input
from keras.models import Model
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Activation, Dropout
from keras.layers import Add, ZeroPadding2D, Flatten

def input_block(x):
    x = ZeroPadding2D(padding=(3,3))(x)
    x = Convolution2D(64, (7,7), strides = (2,2), kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x


def residual_block(x,channels,d_increase=False):
    shortcut = x

    if d_decrease == True:
        stride = (2,2)
    else:
        stride = (1,1) 
  
    
    x = Convolution2D(channels, (1,1), strides=stride, padding = "valid", kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    
    x = Convolution2D(channels, (3,3), padding = "same", kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(4*channels, (1,1), padding = "valid", kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    
        shortcut = Convolution2D(4*channels, (1,1), strides=stride, padding = "valid", kernel_initializer = 'he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)


    x = Add()([x,shortcut])
    x = Activation('relu')(x)

    return x


def block_group(x, channels, num_blocks,d_increase=False):
    for i in range(num_blocks):
        if i==0 :
            x = residual_block(x,channels,d_increase)
        else:
            x = residual_block(x,channels)
    
    return x
