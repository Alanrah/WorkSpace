import os
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def getfilelist(filepath):
    #打印子文件夹名称
    filelist = os.listdir(filepath)
    for num in range(len(filelist)):
        filename = filelist[num]
        if os.path.isdir(filepath + "/" + filename):#判断文件夹是否存在
            #os.chdir(filepath + "/" + filename)
            #如果不为空遍历识别，如果为空删除
            if not os.listdir(filepath + "/" + filename):
                print("文件夹 %s 为空"%(filepath + "/" + filename))
                new = '1-' + filename
                os.rename(filename, new)

path = 'E:/DeepLearning/spiderResult/交换机'#服务器，机柜，路由器，集线器，
os.chdir(path)
getfilelist(path)

