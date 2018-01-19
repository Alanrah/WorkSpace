import os
import numpy as np
from keras import applications
from keras.preprocessing import image
#  预处理 图像编码服从规定，譬如,RGB，GBR这一类的，preprocess_input(x)
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

classesOfServer机架式 = ['projector','desktop_computer','scoreboard','hard_disc','modem','remote_control','radio', 'notebook','computer_keyboard','projector','printer','tape_player','cassette_player','scoreboard','monitor','web_site','harmonica','laptop']

classesOfSwitch = ['modem','radio','safe','tape_player','switch','CD_player','cassette_player','abacus']
classesOfPrinter = ['printer','photocopier','tape_player','cassette_player','CD_player','switch','Polaroid_camera','holster','sewing_machine','projector']
classesOfServer塔式 = ['desktop_computer','hard_disc','binder','modem','switch','radio','notebook','CD_player',
                     'tape_player','cassette_player','space_heater','solar_dish','cash_machine','scoreboard','safe','menu']
classesOfCabinet = ['pay-phone','cash_machine','vending_machine','turnstile','switch','refrigerator','medicine_chest','typewriter_keyboard','modem','monitor','web_site','envelope','Band_Aid','muzzle','whistle','space_bar','desktop_computer',
                    'photocopier','microwave','safe']
classesOfRouter = ['sandal','spatula','swab','rubber_eraser','ballpoint','toilet_seat','goblet','matchstick','lipstick','rubber_eraser','rocking_chair','scale','hook','mosquito_net','modem']

classes = classesOfServer机架式 + classesOfServer塔式
classesServer = list(set(classes))

pic_ext = ['.jpg', '.png']

model = applications.VGG16(weights='imagenet')
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
            #print('Probability %0.2f%% => [%s]' % (100 * result[2], result[1]))
            if result[1]  in classesServer :
                i=1
                break
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
            print("1 %s"%(filepath + "/" + filename))
            getfilelist(filepath + "/" + filename)

        else:
            name, ext = os.path.splitext(filename)
            if ext in pic_ext:
                print("2  %s"%filename)
                VGG(filename)

path = 'E:/DeepLearning/spiderResult/服务器'
os.chdir(path)
getfilelist(path)

