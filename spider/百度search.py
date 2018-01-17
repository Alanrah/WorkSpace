#-*- coding:utf-8 -*-
import re
import requests
import os

def dowmloadPic(html,keyword):
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)
    i = 0
    print ('找到关键词:'+keyword+'的图片，现在开始下载图片...')
    for each in pic_url:
        print ('正在下载第'+str(i+1)+'张图片，图片地址:'+str(each))
        try:
            pic= requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print ('【错误】当前图片无法下载')
            continue
        _path = os.getcwd()
        #new_path = os.path.join(_path , keyword)
        new_path = (_path +  '/image/'+ keyword + '/')

        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        string = new_path + keyword + '_'+str(i) + '.jpg'
        #resolve the problem of encode, make sure that chinese name could be store
        fp = open(string.encode('utf-8'),'wb')
        fp.write(pic.content)
        fp.close()
        i += 1



if __name__ == '__main__':
    #在python3.x中raw_input( )和input( )进行了整合，去除了raw_input( )，仅保留了input( )函数，
    # 其接收任意任性输入，# 将所有输入默认为字符串处理，并返回字符串类型。
    word  ='ThinkServer TS250'
    url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&ct=201326592&v=flip'
    #url = "http://search.jd.com/Search?keyword=ThinkServer%20TS250&enc=utf-8&wq=ThinkServer%20TS250&pvid=2afa427a0aab4725bead47357f82be89"
    #url = 'http://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&word=华为 R4830N&fr=ala&ala=2&alatpl=sp&pos=0'
    result = requests.get(url)
    dowmloadPic(result.text, word )