#coding:utf8
import os
import cv2
import re
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

#https://blog.csdn.net/zh_jnu/article/details/54602262
datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range= 0.2,
        height_shift_range = 0.2,
        rescale = 1.0/255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
'''rotation_range: 旋转范围, 随机旋转(0-180)度;
    width_shift and height_shift: 随机沿着水平或者垂直方向，以图像的长宽小部分百分比为变化范围进行平移;
    rescale: 对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0 - 1之间，通常为1 / 255;
    shear_range: 水平或垂直投影变换, http://keras.io/preprocessing/image/
    zoom_range: 按比例随机缩放图像尺寸;
    horizontal_flip: 水平翻转图像;
    fill_mode: 填充像素, 出现在旋转或平移之后．'''
write_path = "E:/DeepLearning/workSpace/venv/device_img/train/rackServer/"
def eachFile(filepath):
    count = 0
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        print(filepath)
        print(allDir)
        child = os.path.join('%s%s' % (filepath,allDir))
        write_child = os.path.join('%s%s' % (write_path,allDir))
        img = load_img(child)
        nul_num = re.findall(r"\d",child)
        nul_num = int(nul_num[0])
        x = img_to_array(img)
        x = x.reshape((1,)+x.shape)
        i = 0
        for batch in datagen.flow(
                x,
                batch_size =1,
                save_to_dir = write_path,
                save_prefix = nul_num,save_format = 'jpg'):
                count += 1
                i += 1
                if i >= 6 :
                        break
    return count
count = eachFile("E:/DeepLearning/workSpace/venv/device_img/baidu-1500/train/rackServer/")
print ("一共产生了%d张图片"%count)