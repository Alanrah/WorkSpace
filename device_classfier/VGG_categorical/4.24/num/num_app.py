import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
K.set_image_dim_ordering('th')
import os
num_classes = 10
def num_recognition(weights_path=None):
    num_classes = 10
    img_width, img_height = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model

model = num_recognition('6-layers-CNN.h5')
#https://juejin.im/post/5a5882646fb9a01ca871e7d2

filepath="E:/DeepLearning/workSpace/venv/MyWorkSpace/num_classfier/nums/8/"
pathDir =  os.listdir(filepath)
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
for allDir in pathDir:
    child = os.path.join('%s%s' % (filepath, allDir))
    print(child)
    img = image.load_img(child,grayscale=True, target_size=(28, 28))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # (print(x.shape)1, 1, 28, 28)
    proba = model.predict_proba(x, verbose=1)
    result = model.predict_classes(x, verbose=1)
    print('识别为：%s'%result)
    if result==0:
        count0=count0+1
    elif result==1:
        count1=count1+1;
    elif result==2:
        count2=count2+1
    elif result==3:
        count3=count3+1
    elif result==4:
        count4=count4+1
    elif result==5:
        count5=count5+1
    elif result==6:
        count6=count6+1
    elif result==7:
        count7=count7+1
    elif result==8:
        count8=count8+1
    elif result==9:
        count9=count9+1
    else:{}

print(count0,count1,count2,count3,count4,count5,count6,count7,count8,count9)


