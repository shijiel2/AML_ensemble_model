from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import os
from model import resnet_v1, basic_cnn_model
from utils import *


# Training parameters
epochs = 6

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

# Computed depth from supplied model parameter n
n = 5
depth = n * 6 + 2
version = 1

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    # if epoch > 30:
    #     lr *= 1e-2
    # elif epoch > 15:
    #     lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# add model_0
model_0_input = Input(shape=input_shape)
model_0_output = basic_cnn_model(input=model_0_input, depth=depth, num_classes=num_classes, dataset=FLAGS.dataset)[2]
model_0 = Model(input=model_0_input, output=model_0_output)
model_0.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=['accuracy'])
model_0.summary()
model_0.fit(
        x_train, y_train,
        batch_size=FLAGS.batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=1)


model_input = Input(shape=input_shape)
model_dic = {}
model_out = []
for i in range(FLAGS.num_models):
    model_dic[str(i)] = basic_cnn_model(input=model_input, depth=depth, num_classes=num_classes, dataset=FLAGS.dataset)
    model_dic[str(i)][0].set_weights(model_0.get_weights())
    model_out.append(model_dic[str(i)][2])

model_output = keras.layers.concatenate(model_out)

model = Model(input=model_input, output=model_output)

model.compile(
        loss=Loss_withEE_DPP,
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=[acc_metric, Ensemble_Entropy_metric, log_det_metric])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), FLAGS.dataset+'_EE_LED_saved_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)
                        +'_logdetlamda'+str(FLAGS.log_det_lamda)+'_'+str(FLAGS.augmentation))
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_acc_metric', mode='max', verbose=2, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)


callbacks = [checkpoint, lr_scheduler]


# Augment labels
y_train_2 = []
y_test_2 = []
for _ in range(FLAGS.num_models):
    y_train_2.append(y_train)
    y_test_2.append(y_test)
y_train_2 = np.concatenate(y_train_2, axis=-1)
y_test_2 = np.concatenate(y_test_2, axis=-1)



# Run training, with or without data augmentation.
if not FLAGS.augmentation:
    print('Not using data augmentation.')
    model.fit(
        x_train, y_train_2,
        batch_size=FLAGS.batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_2),
        shuffle=True,
        verbose=1,
        callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
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
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
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

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train_2, batch_size=FLAGS.batch_size),
        validation_data=(x_test, y_test_2),
        epochs=epochs,
        verbose=1,
        workers=4,
        callbacks=callbacks)