import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import History
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import keras

def maxIndexOfList(list):
    x = max(list);
    for index,value in enumerate(list):
        if value == x:
            return index;

def vgg_device(weights_path=None):
    img_width, img_height = 150, 150
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model
model = vgg_device('fine_tune_model-adam-lr=0.0001-batch=100.h5')
img = image.load_img('uploads/mmexport1523343801056.jpg', target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(proba)
preds = model.predict(x)
#{'交换机': 0, '塔式服务器': 1, '打印机': 2, '机架式服务器': 3, '机柜': 4, '路由器': 5}
class_index = ['交换机','塔式服务器','打印机','机架式服务器','机柜','路由器']
print(preds[0])
list = preds[0];

index = maxIndexOfList(list);
print("识别结果是：%s"%class_index[index])

# decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率) decode_predictions(y_pred)
