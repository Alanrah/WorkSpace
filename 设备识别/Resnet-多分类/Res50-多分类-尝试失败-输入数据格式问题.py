# coding=utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
# from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
import numpy as np

from keras.utils import plot_model
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
import keras
from keras import backend as K
seed = 7
np.random.seed(seed)

img_width, img_height = 224, 224

train_data_dir = 'E:/DeepLearning/workSpace/venv/设备识别/train'
validation_data_dir = 'E:/DeepLearning/workSpace/venv/设备识别/validations'
nb_train_samples = 9000#1500*6=9000，2000张训练图片构成的数据集，图片名可不遵循规则
nb_validation_samples = 600#100*6=600张验证集
epochs = 10
batch_size =20 #16每一个epoch一共9000/batch_size组

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


inpt = Input(shape=(224, 224, 3))
x = ZeroPadding2D((3, 3))(inpt)
x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(1000, activation='softmax')(x)
#http://keras-cn.readthedocs.io/en/latest/models/model/
model = Model(inputs=inpt, outputs=x)
sgd = SGD(decay=0.0001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
plot_model(model, to_file='Resnet50模型.png',show_shapes=True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


history = History()
model_checkpoint = ModelCheckpoint('temp_model.hdf5', monitor='loss', save_best_only=True,period=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='log', write_images=1, histogram_freq=0)
callbacks = [
        history,
        model_checkpoint,
        tb_cb
    ]

history=model.fit( x=None, y=None, batch_size=20, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=0.0, validation_data=validation_generator, shuffle=True,initial_epoch=0, steps_per_epoch=nb_train_samples // batch_size, validation_steps=nb_validation_samples // batch_size)
model.save('resnet50设备分类.h5')

from matplotlib import pyplot as plt
plt.plot()
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
