import os
import numpy as np
from keras import applications
from keras.preprocessing import image
#  预处理 图像编码服从规定，譬如,RGB，GBR这一类的，preprocess_input(x)
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# 网络上图片的地址
classesOfSwitch = ['modem','radio','safe','tape_player','switch','CD_player','cassette_player','abacus']
classesOfPrinter = ['printer','photocopier','tape_player','cassette_player','CD_player','switch','Polaroid_camera','holster','sewing_machine','projector']
pic_ext = ['.jpg', '.png']
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
            # print('Probability %0.2f%% => [%s]' % (100 * result[2], result[1]))
            if result[1]  in classesOfPrinter :
                i=1
    if i==0:
        newname = '1-'+filename
        os.rename(filename, newname)

def getfilelist(filepath):
    #打印子文件夹名称
    filelist = os.listdir(filepath)
    for num in range(len(filelist)):
        filename = filelist[num]
        if os.path.isdir(filepath + "/" + filename):
            os.chdir(filepath + "/" + filename)
            getfilelist(filepath + "/" + filename)
            print("1 %s"%(filepath + "/" + filename))
        else:
            name, ext = os.path.splitext(filename)
            if ext in pic_ext:
                print("2  %s"%filename)
                VGG(filename)


path = 'E:\DeepLearning\清洗ing\网络设备\打印机'
os.chdir(path)
getfilelist(path)

'''

for file in os.listdir(path):
    if os.path.isfile(file) == True:
        VGG(file)
# 将远程数据下载到本地，第二个参数就是要保存到本地的文件名'''

