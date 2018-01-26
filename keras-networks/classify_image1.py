#来源：https://yq.aliyun.com/articles/78726
#原文：https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/?spm=a2c4e.11153940.blogcont78726.38.3467a8c24UyzhE
# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2 #cv2用于与OpenCV结合

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

#--image为希望进行分类的图像的路径
ap.add_argument("-i", "--image", required=True,help="path to the input image")
#--model为选用的CNN的类别，默认为VGG16
ap.add_argument("-model", "--model", type=str, default="vgg16",help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # TensorFlow ONLY
    "resnet": ResNet50
}# VGG16，VGG19以及ResNet接受224×224的输入图像，Inception V3和Xception要求为299×299

# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")
#输入一个图像到一个CNN中会返回一系列键值，包含标签及对应的概率

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
#初始化预处理函数为keras.preprocess_input
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)#将图像从PIL/Pillow实例转换成NumPy矩阵
# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through thenetwork
# 因为我们往往使用CNN来批量训练/分类图像，所以需要使用np.expand_dims在矩阵中添加一个额外的维度
image = np.expand_dims(image, axis=0)
# pre-process the image using the appropriate function based on the
#  model that has been loaded (i.e., mean subtraction, scaling, etc.)
#使用合适的预处理函数来执行mean subtraction/scaling。
image = preprocess(image)

# classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
#调用.predict函数，并从CNN返回预测值。
preds = model.predict(image)
#decode_predictions函数将预测值解码为易读的键值对：标签、以及该标签的概率。
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
#例如：（"0":  "tench","0.88"）== (imagenetID, label, prob)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
#通过OpenCV从磁盘将输入图像读取出来，在图像上画出最可能的预测值并显示在我们的屏幕上。
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)