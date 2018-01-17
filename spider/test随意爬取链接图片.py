#coding=utf-8

#urllib模块提供了读取Web页面数据的接口
import urllib.request
#re模块主要包含了正则表达式
import re
import os
#定义一个getHtml()函数

_path = os.getcwd()
print(_path)
new_path = os.path.join(_path , 'picture')
if not os.path.isdir(new_path):
    os.mkdir(new_path)
new_path += '/'
print(new_path)

def getHtml(url):
    page = urllib.request.urlopen(url)    #urllib.urlopen()方法用于打开一个URL地址
    html = page.read() #read()方法用于读取URL上的数据
    html = html.decode('utf-8')
    return html

def getImg(html):
    reg = 'src="(.+?\.jpg)" alt='    #正则表达式，得到图片地址
    imgre = re.compile(reg)     #re.compile() 可以把正则表达式编译成一个正则表达式对象.
    imglist = re.findall(imgre,html)      #re.findall() 方法读取html 中包含 imgre（正则表达式）的    数据
    #把筛选的图片地址通过for循环遍历并保存到本地
    #核心是urllib.urlretrieve()方法,直接将远程数据下载到本地，图片通过x依次递增命名
    x = 0

    for imgurl in imglist:
        urllib.request.urlretrieve(imgurl, new_path+'%s.jpg' % x)
        x+=1

html = getHtml("http://item.jd.com/10596564152.html")
print(getImg(html))