from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Model
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
import keras.callbacks
import keras

CLASS_NUM = 6
norm_size = 224

# dimensions of our images.
img_width, img_height = 224, 224
train_data_dir = 'E:/DeepLearning/workSpace/venv/device_img/train/'
validation_data_dir = 'E:/DeepLearning/workSpace/venv/device_img/validations/'
nb_train_samples = 36000  # 1500*6=9000，2000张训练图片构成的数据集，图片名可不遵循规则
nb_validation_samples = 4800  # 100*6=600张验证集
epochs = 2
batch_size = 64  # 16每一个epoch一共9000/batch_size组

train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    #shuffle=True,save_to_dir='E:/DeepLearning/workSpace/venv/设备训练图片/train1',
    #save_prefix=None,
    #save_format='jpg',
    #follow_links=False
     )

#载入数据
def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    dirs = os.listdir(path)
    imagePaths=list(dirs)
    # loop over the input images
    #https://www.cnblogs.com/zangyu/p/5764905.html
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        print("2+%s"%imagePath)
        imagePathx =path+imagePath+'/'
        count = 0
        for filename in os.listdir(imagePathx):
            filename=imagePathx+filename
            image = cv2.imread(filename,cv2.IMREAD_COLOR)
            #cv2.imread()路径中不能带有中文字符
            if image is None:
                os.rename(filename, os.path.join(imagePathx, '000_'+str(count) + ".jpg"))
                continue
            #cv2.imread()导入图片时是BGR通道顺序,转换为RGB模式
            b, g, r = cv2.split(image)
            rgb_img = cv2.merge([r, g, b])
            rgb_img = cv2.resize(rgb_img, (norm_size, norm_size))
            rgb_img = img_to_array(rgb_img)
            count=count+1
            data.append(rgb_img)
            label = imagePath
            label_int =0
            if(label == "cabinet"):
                label_int=0
            elif label=="printer":
                label_int=1
            elif label == "rackServer":
                label_int =2
            elif label == "router":
                label_int = 3
            elif label == "switch":
                label_int = 4
            else :
                label_int = 5
            labels.append(label_int)
        print(count)
        # extract the class label from the image path and update the
        # labels list
    # scale the raw pixel intensities to the range [0, 1]--------------------------------------------------------------------------memoryerror
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    print("4+%s"%labels[0])

    # convert the labels from integers to vectors
    #labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels

testX,testY = load_data(validation_data_dir)
print(testX.shape)
print(testY)

history = History()
model_checkpoint = ModelCheckpoint('vgg19.hdf5', monitor='loss', save_best_only=True, period=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='log', write_images=1, histogram_freq=1)
callbacks = [
    history,
    model_checkpoint,
    tb_cb
]

def self_VGG():
    global history
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    img_input = Input(shape=input_shape)
    #反复堆叠3*3的小型卷积核和2*2的最大池化层
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dense(6, activation='softmax', name='predictions')(x)
    # Create model.
    model = Model(inputs=img_input, outputs=x, name='vgg19')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='vgg19classfier.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # this is the augmentation configuration we will use for training
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(testX,testY),
        validation_steps=nb_validation_samples // batch_size
    )
    model.save_weights('vgg19classfier.h5')
    return model

model = self_VGG()
from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc with adam batch=64 lr=0.001 decay=1')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss with adam batch=64 lr=0.001 decay=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
label = "E:/DeepLearning/workSpace/venv/device_img/train/printer/1.jpg".split('/')[-2]