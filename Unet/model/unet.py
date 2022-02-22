import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model

# U-Net architecture
def unet(input_shape=(1024, 1280, 1), output_channels=2):
    model_in = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(model_in)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    down1 = MaxPooling2D((2, 2), strides=2)(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    down2 = MaxPooling2D((2, 2), strides=2)(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    down3 = MaxPooling2D((2, 2), strides=2)(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    down4 = MaxPooling2D((2, 2), strides=2)(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Expansive path
    up1 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(512, (2, 2), activation='relu', padding='same')(up1)
    concat1 = concatenate([conv4, conv6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = UpSampling2D((2, 2))(conv6)
    conv7 = Conv2D(256, (2, 2), activation='relu', padding='same')(up2)
    concat2 = concatenate([conv3, conv7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = UpSampling2D((2, 2))(conv7)
    conv8 = Conv2D(128, (2, 2), activation='relu', padding='same')(up3)
    concat3 = concatenate([conv2, conv8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = UpSampling2D((2, 2))(conv8)
    conv9 = Conv2D(64, (2, 2), activation='relu', padding='same')(up4)
    concat4 = concatenate([conv1, conv9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(output_channels, (1, 1), activation='softmax', padding='same')(conv9)

    model = Model(model_in, conv10)
    
    return model