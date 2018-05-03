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
import  os
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
model = vgg_device('vgg16_adam_0.001_64.h5')

count = 0
filepath="E:/DeepLearning/workSpace/venv/device_img/baidu-1500/validations/towerServer/"
pathDir =  os.listdir(filepath)
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
for allDir in pathDir:
    child = os.path.join('%s%s' % (filepath, allDir))
    print(child)
    img = image.load_img(child, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    proba = model.predict_proba(x, verbose=0)
    pred = model.predict_classes(x, verbose=0)
    if pred[0]==0:
        count0=count0+1
    elif pred[0]==1:
        count1=count1+1;
    elif pred[0]==2:
        count2=count2+1
    elif pred[0]==3:
        count3=count3+1
    elif pred[0]==4:
        count4=count4+1
    elif pred[0]==5:
        count5=count5+1
    else:{}

print(count0,count1,count2,count3,count4,count5)
'''
    print(pred[0])
    #class_index = ['交换机', '塔式服务器', '打印机', '机架式服务器', '机柜', '路由器']
    #{'towerServer': 5, 'printer': 1, 'rackServer': 2, 'router': 3, 'cabinet': 0, 'switch': 4}
    class_index = ['cabinet', 'printer', 'rackServer', 'router', 'switch', 'towerServer']
    #class_index = ['switch', 'towerServer', 'printer', 'rackServer', 'cabinet', 'router']
    result = class_index[pred[0]]
    print('识别为：')
    print(proba[0])
    print(result)
'''
