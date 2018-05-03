#来源：https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# http://blog.csdn.net/caanyee/article/details/52502759
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
import keras.callbacks
import keras

# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = '../../device_img/5000+1000/train'
validation_data_dir = '../../device_img/5000+1000/validations'
nb_train_samples = 30000  # 1500*6=9000，2000张训练图片构成的数据集，图片名可不遵循规则
nb_validation_samples = 6000  # 100*6=600张验证集
epochs = 100
batch_size = 64  # 16每一个epoch一共9000/batch_size组
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

history = History()
model_checkpoint = ModelCheckpoint('temp_modell.hdf5', monitor='loss', save_best_only=True, period=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='log', write_images=1, histogram_freq=0)
callbacks = [
    history,
    model_checkpoint,
    tb_cb
]

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
print(train_generator.class_indices)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
print(validation_generator.class_indices)
def self_VGG():
    global history
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    #反复堆叠3*3的小型卷积核和2*2的最大池化层
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.summary()
    plot_model(model, to_file='vgg16.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # this is the augmentation configuration we will use for training
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )
    model.save_weights('vgg16_rmsprop_0.001_64.h5')
    return model

model = self_VGG()
from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc with rmsprop batch=64 lr=0.001')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss with rmsprop batch=64 lr=0.001')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()