# coding:utf-8
import requests
from bs4 import BeautifulSoup
import threading
import os
import urllib
import json
import time
from selenium import webdriver

requests.adapters.DEFAULT_RETRIES = 5
url = 'https://item.jd.com/3231165.html'
#driver = webdriver.PhantomJS(executable_path='E:/keraEnv/phantomjs-2.1.1-windows/bin/phantomjs.exe')
driver = webdriver.Chrome(executable_path='E:/keraEnv/anaconda/chromedriver.exe')
driver.get(url)
time.sleep(3)
page = driver.page_source
driver.close()
soup = BeautifulSoup(page,'lxml')
detail = soup.find('div', {'id': 'J-detail-content'})
#print(driver.find_element_by_id('J-detail-content'))
print(detail)
imgs = detail.find_all('img')
for img in imgs :
    src = img.get("data-lazyload")  # 得到已经加载出来的url
    if src:
        fileName = src.split('/')[-1]
        src = 'http:' + src
        print("index 图片： %s" % src)
        urllib.request.urlretrieve(src, filename=fileName)