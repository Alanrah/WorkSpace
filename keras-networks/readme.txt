来源：https://yq.aliyun.com/articles/78726
原文地址：https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/?spm=a2c4e.11153940.blogcont78726.38.3467a8c24UyzhE

安装cv2:http://blog.csdn.net/mumu_1233/article/details/77858950

E:\keraEnv\anaconda\Lib\site-packages>pip install opencv_contrib_python-3.2.0.7-
cp36-cp36m-win_amd64.whl
Processing e:\keraenv\anaconda\lib\site-packages\opencv_contrib_python-3.2.0.7-c
p36-cp36m-win_amd64.whl
Requirement already satisfied: numpy>=1.11.3 in e:\keraenv\anaconda\lib\site-pac
kages (from opencv-contrib-python==3.2.0.7)
Installing collected packages: opencv-contrib-python
Successfully installed opencv-contrib-python-3.2.0.7


E:\keraEnv\anaconda\Lib\site-packages>python
Python 3.6.3 |Anaconda custom (64-bit)| (default, Oct 15 2017, 03:27:45) [MSC v.
1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
ImportError: numpy.core.multiarray failed to import
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "E:\keraEnv\anaconda\Lib\site-packages\cv2\__init__.py", line 7, in <modu
le>
    from . import cv2
ImportError: numpy.core.multiarray failed to import


问题解决：
http://blog.csdn.net/baobao3456810/article/details/52177316

还是失败，numpy有问题



