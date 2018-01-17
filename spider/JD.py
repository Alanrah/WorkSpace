import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlretrieve
import re
from urllib.request import urlopen
class Picture():
    def __init__(self,keyword):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}
        self.base_url ="http://search.jd.com/Search?keyword="+keyword+"&enc=utf-8&wq="+keyword #+"&pvid=2afa427a0aab4725bead47357f82be89"
        # 'https://list.jd.com/list.html?cat=9987,653,655&page='
        self.base_path = os.path.dirname(__file__)

    def makedir(self, name):
        path = os.path.join(self.base_path, name)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("File has been created.")
        else:
            print('OK!The file is existed. You do not need create a new one.')
        os.chdir(path)

    def request(self, url):
        r = requests.get(url, headers=self.headers)
        return r

    def get_img(self,keyword):
        r = self.request(self.base_url )
        hrefs = []
        goodsList = BeautifulSoup(r.text, 'lxml').find('div', id='J_goodsList')
        ul = goodsList.find('ul',{"class":{"gl-warp","clearfix"}})
        pimg =  ul.find_all('div',{'class':'p-img'})
        print(len(pimg))
        aitem = []
        for p in pimg:
            aitem.append(p.find('a'))

        print("一共有 %d 个链接" %len(aitem))
        self.makedir(keyword)
        num = 0
        for i in aitem:
            num += 1
            img = i.find('img')
            href = i.get("href")
            if href != "javascript:;" and href[2] == 'i':
                href ="http:" + href
                hrefs.append(href)
            #else:
                #print(" %d href 不存在"%num)

            if img:
                imgSrc = img.get('src')
                print(imgSrc)
                if imgSrc != None:
                    url = 'http:' + imgSrc
                    fileName = imgSrc.split('/')[-1]
                    urlretrieve(url, filename=fileName)
                    #print('This is %s picture' % num)
                #else:
                    #print(" %d img 不存在" % num)
        return hrefs

def urlImage(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}
    r = requests.get(url,headers = headers)
    bsObject = BeautifulSoup(r.text, 'lxml')
    print(bsObject)
    return
    comments = bsObject.find_all('div',{'class':'comment-item'})
    print("一共 %d 个comment " %len(comments))
    for comment in comments:
        a = comment.find_all('a',{'class':'J-thumb-img'})
        for x in a:
            src = x.find('img').get('src')
            src = convertSrc(src)
            imgUrl = 'http'+src
            fileName = src.split('/')[-1]
            urlretrieve(imgUrl, filename=fileName)

    imgList = bsObject.find_all('img',{"src":re.compile(".*\.jpg$")})
    for img in imgList:
        imgSrc = img.get('src')
        print("imgsrc %s"%imgSrc)
        try:
            if imgSrc:
                if imgSrc[0] != 'h':#有点链接前面有https，有的没有，手动加,OSError: [Errno 22] Invalid argument: '585b94f7N268e3bd5.jpg?t=1491469036387'
                    imgUrl = 'http:' + imgSrc
                fileName = imgSrc.split('/')[-1]
                urlretrieve(imgUrl, filename=fileName)
        except ValueError as e:
            print('ValueError:', e)
        except ZeroDivisionError as e:
            print('ZeroDivisionError:', e)
        except OSError as e:
            print("OSError",e)
        except SyntaxError as e:
            print("SyntaxError",e)
        except AttributeError as e:
            print("AttributeError" , e)

def convertSrc(src):
    tempSrc = src.split('/')
    print(tempSrc)
    curSrc = ''
    num = 0
    for i in tempSrc:
        if num == 3:
            i = 'shaidan'
        if num == 4:
            i = 's616x405_jfs'
        curSrc += '/'
        curSrc += i
        num += 1
    curSrc = '' + curSrc[1:]
    return curSrc

if __name__ == '__main__':
    keyword = "ThinkServer TS250"
    picture = Picture(keyword)
    hrefs = picture.get_img(keyword)
    hrefs = set(hrefs)
    x=0
    for href in hrefs:
        urlImage(href)
        x+=1
        print("爬取第 %d 个页面成功:%s"%(x,href))
