# -*- coding: utf-8 -*
import re
import os
import urllib
from bs4 import BeautifulSoup
import requests
def craw(url, page):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}
    r = requests.get(url, headers=headers)
    goodsList = BeautifulSoup(r.text, 'lxml').find('div', id='J_goodsList')
    ul = goodsList.find('ul', {"class": {"gl-warp", "clearfix"}})
    print(ul)
    imagelist = ul.select(' li > div > div.p-img > a > img')#
    # pricelist=soup.select('#plist > ul > li > div > div.p-price > strong')
    # print pricelist
    path = "E:/{}/".format(str(goods))
    if not os.path.exists(path):
        os.mkdir(path)
    for (imageurl) in zip(imagelist):
        print(imageurl)
        name =  'A'#imageurl.split('/')[-1]
        imagename = path + name + ".jpg"
        imgurl = "http:" + str(imageurl.get('data-lazy-img'))
        if imgurl == 'http:None':
            imgurl = "http:" + str(imageurl.get('src'))
        try:
            urllib.request.urlretrieve(imgurl, filename=imagename)
            print("")
        except:
            continue

'''
#J_goodsList > ul > li:nth-child(1) > div > div.p-img > a > img
#plist > ul > li:nth-child(1) > div > div.p-name.p-name-type3 > a > em
#plist > ul > li:nth-child(1) > div > div.p-price > strong:nth-child(1) > i
'''

if __name__ == "__main__":
    goods = 'ThinkServer TS250'
    pages = 1
    count = 0.0
    for i in range(1, pages + 1, 2):
        url = "https://search.jd.com/Search?keyword={}&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&suggest=1.def.0.T06&wq=diann&page={}".format(str(goods), str(i))
        print(url)
        craw(url, i)
        count += 1
        print('work completed {:.2f}%'.format(count / pages * 100))