import requests
from bs4 import BeautifulSoup
import threading
import os
import urllib
import json
import time

requests.adapters.DEFAULT_RETRIES = 5
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
#  预处理 图像编码服从规定，譬如,RGB，GBR这一类的，preprocess_input(x)
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


# 网络上图片的地址
classes = ['modem','radio','safe','tape_player','switch','CD_player','cassette_player','abacus']
def VGG(filename):
    model = applications.VGG16(weights='imagenet')
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    i=0
    # decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率) decode_predictions(y_pred)
    for results in decode_predictions(preds):
        for result in results:
            print('Probability %0.2f%% => [%s]' % (100 * result[2], result[1]))
            if result[1]  in classes :
                i=1
    if i==0:
        newname = '1-'+filename
        os.rename(filename, newname)

img_src = 'http://img.my.csdn.net/uploads/201212/25/1356422284_1112.jpg'
path = 'E:/DeepLearning/workSpace/venv/VGG'
filename = '1.jpg'
os.chdir(path)
# 将远程数据下载到本地，第二个参数就是要保存到本地的文件名
urllib.request.urlretrieve(img_src, filename)
VGG(filename)