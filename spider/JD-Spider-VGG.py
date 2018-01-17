import requests
from bs4 import BeautifulSoup
import threading
import os
import urllib
import json
import time
requests.adapters.DEFAULT_RETRIES = 5
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
#  预处理 图像编码服从规定，譬如,RGB，GBR这一类的，preprocess_input(x)
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
class spiders:
    def __init__(self, page,keyword):
        self.url = 'https://search.jd.com/Search?keyword='+keyword+'&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&offset=5&wq='+keyword+'&page=' + str(page)
        self.headers = {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        self.search_urls = 'https://search.jd.com/s_new.php?keyword='+keyword+'&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&offset=3&wq='+keyword+'&page={0}&s=26&scrolling=y&pos=30&show_items={1}'
        self.pids = set()  # 页面中所有的id,用来拼接剩下的30张图片的url,使用集合可以有效的去重
        self.img_urls = set()  # 得到的所有图片的url
        self.search_page = page + 1  # 翻页的作用
        self.base_path = os.path.dirname(__file__)+'/网络设备/交换机/'
        self.keyword = keyword
        self.ahrefs = set()
        self.tmp = ''
        self.current_path = self.base_path
        self.classes = ['modem','radio','safe','tape_player','switch','CD_player','cassette_player','abacus']

    def makedir(self, name):
        path = os.path.join(self.base_path, name)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print(" %s File has been created."%path)
        self.current_path = path + '/'
        os.chdir(path)

    # 得到每一页的网页源码
    def get_html(self):
        try:
            res = requests.get(self.url, headers=self.headers)
            html = res.text
            res.close()
            return html
        except:
            time.sleep(3)

    # 得到每一个页面的id，图片链接页面的id：href="//item.jd.com/10329818949.html"
    def get_pids(self):
        try:
            html = self.get_html()
            soup = BeautifulSoup(html, 'lxml')
            lis = soup.find_all("li", class_='gl-item')
            for li in lis:
                data_pid = li.get("data-sku")
                if (data_pid):
                    self.pids.add(data_pid)
                    item = ahref(data_pid)
                    self.ahrefs.add(item)
                    self.tmp = data_pid
        except:
            time.sleep(3)

    # 得到每一个页面的图片和一些数据，由于这是aiax加载的，因此前面一段的img属性是src，后面的属性是data-lazy-img
    def get_src_imgs_data(self):
        try:
            html = self.get_html()
            soup = BeautifulSoup(html, 'lxml')
            divs = soup.find_all("div", class_='p-img')  # 图片
            # divs_prices = soup.find_all("div", class_='p-price')   #价格
            for div in divs:
                img_1 = div.find("img").get('data-lazy-img')  # 得到没有加载出来的url，懒加载图片
                img_2 = div.find("img").get("src")  # 得到已经加载出来的url
                if img_1:
                    self.img_urls.add(img_1)
                    fileName =  img_1.split('/')[-1]
                    img_1 = 'http:' + img_1
                    print("内： %s" % img_1)
                    urllib.request.urlretrieve(img_1, filename=fileName)
                    self.VGG(fileName)
                if img_2:
                    self.img_urls.add(img_2)
                    fileName = img_2.split('/')[-1]
                    img_2 = 'http:' +img_2
                    print("内： %s" % img_2)
                    urllib.request.urlretrieve(img_2, filename=fileName)
                    self.VGG(fileName)
        except:
            time.sleep(3)


    # 这个是得到后面30张的图片和数据，由于是ajax加载的，打开一页会显示前30张的一部分，但是后面30张都保存在这个网页中，因此要请求这个网页得到原来的网站
    def get_extend_imgs_data(self):
        self.search_urls = self.search_urls.format(str(self.search_page), ','.join(self.pids))
        # 拼凑url,将获得的单数拼成url,其中show_items中的id是用','隔开的，因此要对集合中的每一个id分割，page就是偶数，这里直接用主网页的page加一就可以了
        print("外部 URL： %s"%self.search_urls)
        try:
            html = requests.get(self.search_urls, headers=self.headers).text
            soup = BeautifulSoup(html, 'lxml')
            div_search = soup.find_all("div", class_='p-img')
            for div in div_search:
                img_3 = div.find("img").get('data-lazy-img')
                img_4 = div.find("img").get("src")

                if img_3:
                    self.img_urls.add(img_3)
                    fileName = img_3.split('/')[-1]
                    img_3 = 'http:' + img_3
                    print ("外： %s"%img_3)
                    urllib.request.urlretrieve(img_3, filename=fileName)
                    self.VGG(fileName)
                if img_4:
                    self.img_urls.add(img_4)
                    fileName = img_4.split('/')[-1]
                    img_4 = 'http:' + img_4
                    print("外： %s" % img_4)
                    urllib.request.urlretrieve(img_4, filename=fileName)
                    self.VGG(fileName)
        except:
            time.sleep(3)

    # 具体商品页面，可抓取评论图片
    # 京东商品评论信息是由JS动态加载的，所以直接抓取商品详情页的URL并不能获得商品评论的信息。
    # 因此我们需要先找到存放商品评论信息的文件。
    def get_ahref_imgs_data(self):
        fp = open("result.txt", "a+")
        for href in self.ahrefs:
            fp.write(str(href) + '\r\n')
        fp.close()

    #爬取评论
    def crawl(self):
        #request headers
        header = {"user-agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
                  "accept":"*/*",
                  "accept-encoding":"gzip, deflate, br",
                  "accept-language":"zh-CN,zh;q=0.9",
                  "cache-control":"no-cache",
                  "referer":"https://item.jd.com/",
                  }
        cookie = {'__jda':'122270672.1962510787.1512695539.1514518204.1514855059.18',
                '__jdb':'122270672.10.1962510787|18.1514855059',
                '__jdc':'122270672',
                '__jdu':'1962510787',
                '__jdv':'baidu-pinzhuan |122270672|t_288551095_baidupinzhuan|cpc',
                'cn':'1',
                '3AB9D23F7A4B3C9B':'UDZM3N3YIBE3YQ5M5QDEFUSGSALWCMUZHEP52HIBFCJM52CJXI5WB4PUXVAE275RTKBJGBUNEB6TX6SB5D4IMCTXYU',
                'PCSYCityID':'1',
                'ipLoc-djd':'1-72-2799-0',
                'mt_xid':'V2_52007VwMaVV5cUlsZQB9sDGcKQQEOXFNGGRwaDBliAUBXQVFUXxtVGAlQZgcSUl1eAFMYeRpdBWEfElFBW1NLH0gSWAZsABRiX2hSahZAGVwAZgATUFhcUVsWTR1aB2MzEVBeWw%3D%3D',
                'user-key':'b3f81a7e-1518-4635-b183-2c9757b42c56',}
        if len(self.pids)>0:
            p = 0
            purl = 'https://sclub.jd.com/comment/productPageComments.action?&productId=' +self.tmp + '&score=0&sortType=5&page=' + str(
                p) + '&pageSize=10&isShadowSku=0&fold=1'
            pr = requests.get(url=purl, headers=header, cookies=cookie)
            pc = json.loads(pr.text)
            maxPage = pc['maxPage']

            for id in self.pids:
                for p in range(0,maxPage):
                    url = 'https://sclub.jd.com/comment/productPageComments.action?&productId=' + id + '&score=0&sortType=5&page=' + str(p) + '&pageSize=10&isShadowSku=0&fold=1'
                    try:
                        r = requests.get(url=url, headers=header, cookies=cookie)
                        c = json.loads(r.text)
                        tmp = c['comments']
                        print("正在抓取 %s  pid 为 %s 的第 %d 页"%(self.keyword,id,p))
                        path = self.current_path + id #
                        path =path + '/'
                        list_get(tmp,self.current_path)
                    except :
                        pass
        return


    def main(self):
        self.makedir(self.keyword)
        self.get_pids()
        self.get_src_imgs_data()
        self.get_extend_imgs_data()
        self.get_ahref_imgs_data()
        self.crawl()

    def VGG(self,filename):
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
                if result[1]  in self.classes :
                    i=1
        if i==0:
            newname = '1-'+filename
            os.rename(filename, newname)

def list_get(tmp,path):
    #print(type(tmp)) list
    #print(type(tmp[1])) dict
    #print(type(showOrderComment)) dict
    #print(type(content)) str
    if len(tmp)==0:
        print("该页面没有评论")
        return
    for i in tmp:
        if ('showOrderComment' in i.keys()):
            showOrderComment = i['showOrderComment']
            content = showOrderComment['content']
            soup = BeautifulSoup(content, 'lxml')
            imgs =  soup.find_all('img')
            if(len(imgs)==0):
                return
            for img in imgs:
                src = img.get('src')
                if not src.startswith('http:'):
                    src = 'http:{}'.format(src)
                print()
                fileName = src.split('/')[-1]
                fileName = path + fileName
                try:
                    r = requests.get(src)
                    with open(fileName, 'wb') as f:
                        f.write(r.content)
                        f.flush()
                        f.close()
                    print('comment %s success'%src)
                    spider.VGG(fileName)
                    r.close()
                except IOError as e :
                    print(e)
                except:
                    print('comment %s error'%src)
        else:
            print("该评论无图片")
    return

def ahref (id):
    return ('https://item.jd.com/'+id+'.html')

if __name__ == '__main__':
    threads = []

    keywordsSwitch = ['TP-LINK SG1016DT 16口千兆交换机']#,'TP-LINK TL-SG1008D 8口千兆交换机'
    for j in keywordsSwitch:
        for i in range(1, 10):
            page = i * 2 - 1
            spider = spiders(page,j)
            spider.main()
        print(" ---------------------%s 爬取完毕------------------" % j)


    '''
    for j in keywords:
        for i in range(1, 10):
            page = i * 2 - 1  # 这里每一页对应的都是奇数，但是ajax的请求都是偶数的，所有在获取扩展的网页时都要用page+1转换成偶数
            t = threading.Thread(target=spiders(page,j).main, args=[])
            threads.append(t)
        for t in threads:
            t.start()
            t.join()
        print (" %s end"%j)'''
