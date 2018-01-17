import numpy as np
#keras的图像预处理(single/batch)http://keras-cn.readthedocs.io/en/latest/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from http://keras-cn.readthedocs.io/en/latest/other/application/

#http://keras-cn.readthedocs.io/en/latest/utils/get_file()
#应用的时候，需要下载.h5文件，有时候会网络错误，可以提前按照官网提示下载到D:\DeepLearning\Anaconda\Lib\site-packages\keras\datasets
model = applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
img = image.load_img('机柜.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
# decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率) decode_predictions(y_pred)
for results in decode_predictions(preds):
    for result in results:
        print('Probability %0.2f%% => [%s]' % (100*result[2], result[1]))
