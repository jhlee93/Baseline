import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import imageio
import glob
import json
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from model.unet import unet
from dataloader import data_load
print(tf.test.is_gpu_available())

# Learning rate scheduler
def scheduler(epoch, lr):
    warmup_epoch = int(epoch*0.1)
    min_lr = 1e-6
    
    if epoch < warmup_epoch:
        return lr
    else:
        update_lr = lr * tf.math.exp(-0.1)
        if update_lr < min_lr:
            return min_lr
        else:
            return update_lr

# Set path
train_image_path = './dataset/train'
test_image_path = './dataset/test'
train_annot_path = './dataset/train/annotation_total.json'
test_annot_path = './dataset/test/annotation_total.json'

x_train, y_train = data_load(train_annot_path, train_image_path)
x_test, y_test = data_load(test_annot_path, test_image_path)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)

out_channels = np.unique(y_train).shape[0] # 0 or 1
print('output channels:', out_channels)

# # Sample
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
# ax1.imshow(x_train[5], cmap='gray')
# ax2.imshow(y_train[5], cmap='gray')
# plt.show()
# plt.close('all')

model = unet(input_shape=x_train.shape[1:], output_channels=out_channels)
model.summary()

weights_dir = './weights'
log_dir = './logs'
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Set parm
max_epochs = 100

my_callbacks = [
    EarlyStopping(monitor='val_loss', patience=20),
    ModelCheckpoint(f'{weights_dir}/best.h5', save_best_only=True),
    LearningRateScheduler(scheduler),
    TensorBoard(log_dir=log_dir)
]

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x=x_train, y=y_train,
    validation_data=(x_test, y_test),
    batch_size=2,
    epochs=80,
    verbose=True,
    callbacks=my_callbacks
)




