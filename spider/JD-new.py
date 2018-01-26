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


class spiders:
    def __init__(self, page,keyword):
        s = (page//2) *60 +1
        #psort=4&
        self.url = 'https://search.jd.com/Search?keyword=' + keyword + '&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=' + keyword + '&page=' + str(
            page)+'&s='+str(s)+'&psort=4&click=0'
        self.headers = {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        self.search_urls = 'https://search.jd.com/s_new.php?keyword=' + keyword + '&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&offset=3&wq=' + keyword + '&page={0}&s=26&scrolling=y&pos=30&show_items={1}'
        self.pids = set()  # 页面中所有的id,用来拼接剩下的30张图片的url,使用集合可以有效的去重
        self.img_urls = set()  # 得到的所有图片的url
        self.search_page = page + 1  # 翻页的作用
        self.base_path = 'E:/DeepLearning/spiderResult' + '/'
        self.keyword = keyword
        self.ahrefs = {}
        self.tmp = ''
        self.current_path = self.base_path
        self.names = {}

    def makedir(self, name):
        path = os.path.join(self.base_path, name)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print(" %s File has been created." % path)
        self.current_path = path + '/'
        os.chdir(path)
        print(path)

    def get_html(self):
        try:
            res = requests.get(self.url, headers=self.headers)
            html = res.content
            res.close()
            return html
        except:
            time.sleep(10)

    # 得到每一个页面的id，图片链接页面的id：href="//item.jd.com/10329818949.html"
    def get_pids(self):
        try:
            html = self.get_html()
            soup = BeautifulSoup(html, 'lxml')
            lis = soup.find_all("li", class_='gl-item')
            for li in lis:
                data_pid = li.get("data-sku")
                #得到该pid对应的goodname
                div =li.find("div",class_="p-name p-name-type-2")
                aname = div.find('a')
                name =aname.em.text
                if (data_pid):
                    self.pids.add(data_pid)
                    self.names[data_pid] = name
                    item = ahref(data_pid)
                    self.ahrefs[data_pid] =item

        except:
            time.sleep(10)

    def get_index(self,url):
        try:
            driver = webdriver.Chrome(executable_path='E:/keraEnv/anaconda/chromedriver.exe')
            driver.get(url)
            time.sleep(3)
            page = driver.page_source
            driver.close()
            soup = BeautifulSoup(page, 'lxml')
            detail = soup.find('div', {'id': 'J-detail-content'})
            imgs = detail.find_all('img')
            tables = detail.find_all('table')
            i=1
            j=1
            for img in imgs:
                src = img.get("data-lazyload")  # 得到已经加载出来的url
                if src:
                    fileName = str(i)+'.jpg'#src.split('/')[-1]
                    i=i+1
                    src = 'http:' + src
                    print("index 图片： %s" % src)
                    urllib.request.urlretrieve(src, filename=fileName)
            if len(tables)>0:
                for table in tables:
                    src = table.get('background')
                    if src:
                        fileName = str(i)+'.jpg'#src.split('/')[-1]
                        j=j+1
                        src = 'http:' + src
                        print("index 图片： %s" % src)
                        urllib.request.urlretrieve(src, filename=fileName)

        except:
            time.sleep(3)

        #爬取评论
    def crawlComments(self):
        # request headers
        header = {
            "user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "referer": "https://item.jd.com/",
            }
        cookie = {'__jda': '122270672.1962510787.1512695539.1514518204.1514855059.18',
                  '__jdb': '122270672.10.1962510787|18.1514855059',
                  '__jdc': '122270672',
                  '__jdu': '1962510787',
                  '__jdv': 'baidu-pinzhuan |122270672|t_288551095_baidupinzhuan|cpc',
                  'cn': '1',
                  '3AB9D23F7A4B3C9B': 'UDZM3N3YIBE3YQ5M5QDEFUSGSALWCMUZHEP52HIBFCJM52CJXI5WB4PUXVAE275RTKBJGBUNEB6TX6SB5D4IMCTXYU',
                  'PCSYCityID': '1',
                  'ipLoc-djd': '1-72-2799-0',
                  'mt_xid': 'V2_52007VwMaVV5cUlsZQB9sDGcKQQEOXFNGGRwaDBliAUBXQVFUXxtVGAlQZgcSUl1eAFMYeRpdBWEfElFBW1NLH0gSWAZsABRiX2hSahZAGVwAZgATUFhcUVsWTR1aB2MzEVBeWw%3D%3D',
                  'user-key': 'b3f81a7e-1518-4635-b183-2c9757b42c56', }
        if len(self.pids) > 0:
            for id in self.pids:
                p = 1
                purl = 'https://sclub.jd.com/comment/productPageComments.action?&productId=' + id + '&score=0&sortType=5&page=' + str(
                    p) + '&pageSize=10&isShadowSku=0&fold=1'
                pr = requests.get(url=purl, headers=header, cookies=cookie)
                pc = json.loads(pr.text)
                maxPage = pc['maxPage']

                strxxx = self.names[id].replace('/','')#防止文件夹嵌套
                strx = strxxx.replace('*','_')
                ss = strx.replace('|','_')
                sss = ss.replace(':','_')
                ssss = sss.replace('?','_')
                strxx = ssss.split(' ')#防止地址过长
                if(len(strxx)>=5):
                    join_str = strxx[0] + ' ' + strxx[1] + ' ' + strxx[2] + ' ' + strxx[3] + ' ' + strxx[4]
                t_path = self.current_path + join_str
                print('t_path %s '%t_path)
                # 如果最后一个字符是空格
                if t_path[-1] == ' ':
                    t_path = t_path[:-1]
                isExist = os.path.exists(t_path)
                if not isExist:
                    os.makedirs(t_path)
                    print(" %s File has been created." % t_path)
                t_path = t_path+'/'
                os.chdir(t_path)
                print(t_path)
                return

                #爬取该item主页
                self.get_index(self.ahrefs[id])

                #爬取该item的comments
                for p in range(1, maxPage):
                    url = 'https://sclub.jd.com/comment/productPageComments.action?&productId=' + id + '&score=0&sortType=5&page=' + str(
                        p) + '&pageSize=10&isShadowSku=0&fold=1'
                    try:
                        r = requests.get(url=url, headers=header, cookies=cookie)
                        c = json.loads(r.text)
                        tmp = c['comments']
                        print("正在抓取 %s  pid 为 %s 的第 %d 页comments" % (self.names[id], id, p))
                        list_get(tmp, t_path)
                    except:
                        pass
        return
#不知道为啥请求结果是：{'referenceId': 4122232, 'imgComments': {'imgList': None, 'imgCommentCount': 0}}
    def crawlCommentsImage(self):
        # request headers
        header = {
            "user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "referer": "https://item.jd.com/",
            }
        cookie = {'__jda': '122270672.1962510787.1512695539.1514518204.1514855059.18',
                  '__jdb': '122270672.10.1962510787|18.1514855059',
                  '__jdc': '122270672',
                  '__jdu': '1962510787',
                  '__jdv': 'baidu-pinzhuan |122270672|t_288551095_baidupinzhuan|cpc',
                  'cn': '1',
                  'unpl':'V2_ZzNtbUBQERd9AUNVLxAMA2JXFApLAktCcgtBXXIbXFduBBJcclRCFXwUR1RnGFQUZgsZXUBcQBVFCHZXchBYAWcCGllyBBNNIEwHDCRSBUE3XHxcFVUWF3RaTwEoSVoAYwtBDkZUFBYhW0IAKElVVTUFR21yVEMldQl2VHMeXwFnAhRURGdzEkU4dlF5G14GYTMTbUNnAUEpD0FQcx1YSGcLFV5GV0ITfA52VUsa',
                  '3AB9D23F7A4B3C9B': 'UDZM3N3YIBE3YQ5M5QDEFUSGSALWCMUZHEP52HIBFCJM52CJXI5WB4PUXVAE275RTKBJGBUNEB6TX6SB5D4IMCTXYU',
                  'PCSYCityID': '1',
                  'ipLoc-djd': '1-72-2799-0',
                  'mt_xid': 'V2_52007VwMaVV5cUlsZQB9sDGcKQQEOXFNGGRwaDBliAUBXQVFUXxtVGAlQZgcSUl1eAFMYeRpdBWEfElFBW1NLH0gSWAZsABRiX2hSahZAGVwAZgATUFhcUVsWTR1aB2MzEVBeWw%3D%3D',
                  'user-key': 'b3f81a7e-1518-4635-b183-2c9757b42c56',
                  'ipLocationb':'%u5317%u4EAC'}
        if len(self.pids) > 0:
            for id in self.pids:
                p = 1
                purl = 'https://club.jd.com/discussion/getProductPageImageCommentList.action?productId='+id+'&isShadowSku=0&&page='+ str(
                    p) +'&pageSize=10'
                pr = requests.get(url=purl, headers=header, cookies=cookie)
                pc = json.loads(pr.text)
                imgComments = pc['imgComments']
                imgCommentCount = imgComments['imgCommentCount']

                strxxx = self.names[id].replace('/', '')  # 防止文件夹嵌套
                strx = strxxx.replace('*', '_')
                ss = strx.replace('|', '_')
                sss = ss.replace(':', '_')
                ssss = sss.replace('?', '_')
                strxx = ssss.split(' ')  # 防止地址过长
                join_str = ''
                if (len(strxx) >= 5):
                    join_str = strxx[0] + ' ' + strxx[1] + ' ' + strxx[2] + ' ' + strxx[3] + ' ' + strxx[4]
                else:
                    join_str = ssss
                t_path = os.path.join(self.current_path, join_str)
                if t_path[len(t_path)-1] == ' ':
                    t_path = t_path[:-1]
                isExist = os.path.exists(t_path)
                if not isExist:
                    os.makedirs(t_path)
                    print(" %s File has been created." % t_path)
                t_path = t_path+'/'
                os.chdir(t_path)
                print(t_path)

                #爬取该item主页
               # self.get_index(self.ahrefs[id])

                #爬取该item的comments
                maxPage = imgCommentCount//10+1
                for p in range(1,maxPage ):
                    url = 'https://club.jd.com/discussion/getProductPageImageCommentList.action?productId='+id+'&isShadowSku=0&&page='+ str(
                    p) +'&pageSize=10'
                    try:
                        r = requests.get(url=url, headers=header, cookies=cookie)
                        c = json.loads(r.text)
                        imgComment = c['imgComments']
                        imgList = imgComment['imgList']
                        print("正在抓取 %s  pid 为 %s 的第 %d 页comments" % (self.names[id], id, p))
                        for img in imgList:
                            if img['imageUrl']:
                                src = img['imageUrl']
                                if not src.startswith('http:'):
                                    src = 'http:{}'.format(src)
                                fileName = src.split('/')[-1]
                                fileName = t_path + fileName
                                try:
                                    r = requests.get(src)
                                    with open(fileName, 'wb') as f:
                                        f.write(r.content)
                                        f.flush()
                                        f.close()
                                    print('comment img %s success' % src)
                                    r.close()
                                except IOError as e:
                                    print(e)
                                except:
                                    print('comment %s error' % src)

                    except:
                        pass
        return

    def main(self):
        self.makedir(self.keyword)
        self.get_pids()
        self.crawlCommentsImage()


def list_get(tmp,path):
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
                print("该页无评论")
                return
            for img in imgs:
                src = img.get('src')
                if not src.startswith('http:'):
                    src = 'http:{}'.format(src)
                fileName = src.split('/')[-1]
                fileName = path + fileName
                try:
                    r = requests.get(src)
                    with open(fileName, 'wb') as f:
                        f.write(r.content)
                        f.flush()
                        f.close()
                    print('show comment %s success'%src)
                    r.close()
                except IOError as e :
                    print(e)
                except:
                    print('comment %s error'%src)
        else:
            print("该评论无图片")
            #下面这段功能和上面一样

        '''        if ('images' in i.keys()):
            images = i['images']
            for img in images:
                src = img['imgUrl']
                src = convertSrc(src)
                if not src.startswith('http:'):
                    src = 'http:{}'.format(src)
                fileName = src.split('/')[-1]
                fileName = '_'+fileName#避免这些图片地址转换错误
                fileName = path + fileName
                try:
                    r = requests.get(src)
                    with open(fileName, 'wb') as f:
                        f.write(r.content)
                        f.flush()
                        f.close()
                    print('images comment %s success' % src)
                    r.close()
                except IOError as e:
                    print(e)
                except:
                    print('comment %s error' % src)
            else:
                print("该评论无图片")'''
    return

def convertSrc(src):
    tempSrc = src.split('/')
    curSrc = ''
    num = 0
    for i in tempSrc:
        if num == 3:
            i = 'shaidan'
        if num == 4:
            i = 'jfs'
        curSrc += '/'
        curSrc += i
        num += 1
    curSrc = '' + curSrc[1:]
    return curSrc
def ahref (id):
    return ('https://item.jd.com/'+id+'.html')

keywordsCabinet = ['机架式服务器 惠普']

for j in keywordsCabinet:
    print(j)
    for i in range(1, 30):
        page = i * 2 - 1
        print(i)
        spider = spiders(page,j)
        spider.main()

    print(" ---------------------%s 爬取完毕------------------" % j)