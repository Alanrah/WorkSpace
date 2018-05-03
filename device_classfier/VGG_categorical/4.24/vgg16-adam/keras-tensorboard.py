from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import os
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
from sklearn.metrics import roc_auc_score
import numpy as np
import keras

CLASS_NUM = 6
norm_size = 224

img_width, img_height = 224, 224
train_data_dir = '../../device_img/train-roc/'
validation_data_dir = '../../device_img/validation-roc/'
nb_train_samples = 6000  # 1500*6=9000，2000张训练图片构成的数据集，图片名可不遵循规则
nb_validation_samples = 1799  # 100*6=600张验证集
epochs = 2
batch_size = 32  # 16每一个epoch一共9000/batch_size组
'''
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
'''
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
    return data, labels
trainX,trainY=load_data(train_data_dir)
testX,testY = load_data(validation_data_dir)
print(trainX.shape)
print(testX.shape)


class roc_callback(keras.callbacks.Callback):
    def __init__(self, trainX,trainY, testX,testY):
        self.x = trainX
        self.y = trainY
        self.x_val = testX
        self.y_val = testY
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0:2])
        yp = []
        for i in range(0, len(y_pred)):
            yp.append(y_pred[i][0])
        yt = []
        for x in self.validation_data[2]:
            yt.append(x[0])
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)
        print('val-loss', logs.get('loss'), ' val-auc: ', auc)
        print('\n')
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return


callbacks = [roc_callback(trainX,trainY, testX,testY)]
model_checkpoint = ModelCheckpoint('keras-tensorboard1.hdf5', monitor='loss', save_best_only=True, period=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='log', write_images=1, histogram_freq=0)
'''callbacks = [
    history,
    model_checkpoint,
    tb_cb
]'''

def self_VGG():
    global history
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    # 反复堆叠3*3的小型卷积核和2*2的最大池化层
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
    plot_model(model, to_file='keras-tensorboard.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    histories = Histories()
    model.fit(
        trainX, trainY,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[histories, model_checkpoint, tb_cb],
        validation_data=(testX,testY),
        validation_steps=nb_validation_samples // batch_size
    )
    model.save_weights('keras-tensorboard.h5')
    return model

model = self_VGG()
from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc with adam batch=32 lr=0.001 decay=1')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss with adam batch=32 lr=0.001 decay=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()