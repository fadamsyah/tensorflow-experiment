import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Add
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast, RandomCrop
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation, RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomHeight, RandomWidth

# Augmentation Layer
def AugLayer(resize=None, rescaling=None, contrast=None, crop=None,
             flip=None, rotation=None, translation=None, zoom=None,
             height=None, width=None):
    
    aug_layer = Sequential()
    
    if resize: aug_layer.add(Resizing(*resize, interpolation='lanczos3'))
    if rescaling: aug_layer.add(Rescaling(rescaling))
    if contrast: aug_layer.add(RandomContrast(contrast))
    if crop: aug_layer.add(RandomCrop(*crop))
    if flip: aug_layer.add(RandomFlip(flip))
    if rotation: aug_layer.add(RandomRotation(rotation/360.))
    if translation: aug_layer.add(RandomTranslation(*translation))
    if zoom: aug_layer.add(RandomZoom(zoom))
    if height: aug_layer.add(RandomHeight(height))
    if width: aug_layer.add(RandomWidth(width))
    
    return aug_layer

# Classification Model for Low-Res Input
def ClassificationModel(input_shape, n_class, aug_layer=None):
    X_input = Input(input_shape, name='input')
    
    if aug_layer:
        X = aug_layer(X_input)
        X = BatchNormalization(name='pre_bn')(X)
    else:
        X = BatchNormalization(name='pre_bn')(X_input)
    
    X = ZeroPadding2D((1,1), name='zero_pad_0')(X)
    X = MaxPool2D((2,2), strides=2, padding="valid", name='max_pool_0')(X)
    
    X = ZeroPadding2D((1,1), name='zero_pad_1')(X)
    X = Conv2D(64, (3,3), strides=(1,1), padding="valid", use_bias=False, name='conv_1')(X)
    X_shortcut = BatchNormalization(name='bn_1')(X)
    
    X = ZeroPadding2D((1,1), name='zero_pad_2_0')(X_shortcut)
    X = Conv2D(64, (3,3), strides=(1,1), padding="valid", use_bias=False, name='conv_2_0')(X)
    X = BatchNormalization(name='bn_2_0')(X)
    X = Activation('relu', name='relu_2_0')(X)
    X = ZeroPadding2D((1,1), name='zero_pad_2_1')(X)
    X = Conv2D(64, (3,3), strides=(1,1), padding="valid", use_bias=False, name='conv_2_1')(X)
    X = BatchNormalization(name='bn_2_1')(X)
    X = Add(name='skip_connection_1')([X_shortcut, X])
    X = Activation('relu', name='relu_2_1')(X)
    
    X = AveragePooling2D((2,2), strides=(2,2), name='avg_pool')(X)
    
    X = Flatten(name='flatten')(X)
    X = Dropout(0.45, name='dropout_1')(X)
    X = Dense(256, activation='relu', name='dense_layer')(X)
    X = Dropout(0.45, name='dropout_2')(X)
    if n_class == 1: X = Dense(1, activation='sigmoid', name='output_layer')(X)
    else: X = Dense(n_class, activation='softmax', name='output_layer')(X)
    
    return Model(inputs = X_input, outputs = X, name = "Classifier")

# Checkpoint Callback
def CheckpointCallback(filepath, save_weight_only=False, save_best_only=True, monitor='val_acc', mode='auto'):
    return ModelCheckpoint(filepath=filepath, save_weight_only=save_weight_only,
                           save_best_only=save_best_only, monitor=monitor,
                           mode=mode)
    
# Custom learning rate schedule for training
def CustomLRDecay(arr_lr, arr_epoch, verbose=1):
    def scheduler(epoch, lr):
        for i in range(len(arr_epoch)):
            if epoch < arr_epoch[i]:
                break
        return arr_lr[i]
    return LearningRateScheduler(scheduler, verbose)

# plot training and validation history
def plot_history(history, target_name=None, figsize=(16,9), dpi=300, transparent=False, show=True, save=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Loss vs. epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Training', 'Validation'], loc='upper right')
    
    ax2.plot(history.history['acc'])
    ax2.plot(history.history['val_acc'])
    ax2.set_title('Accuracy vs. epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Training', 'Validation'], loc='upper right')
    
    if save: fig.savefig(target_name, dpi=dpi, transparent=transparent)
    if show: plt.show()
