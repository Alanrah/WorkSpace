import os
def getfilelist(filepath, tabnum=1):
    #打印子文件夹名称
    simplepath = os.path.split(filepath)[1]
    returnstr = simplepath + "menu<>" + "\n"
    returndirstr = ""
    returnfilestr = ""
    #查找叶子文件
    filelist = os.listdir(filepath)
    for num in range(len(filelist)):
        filename = filelist[num]
        if os.path.isdir(filepath + "/" + filename):
            returndirstr += "\t" * tabnum + getfilelist(filepath + "/" + filename, tabnum + 1)
        else:
            returnfilestr += "\t" * tabnum + filename + "\n"
    returnstr += returnfilestr + returndirstr
    return returnstr + "\t" * tabnum + "</>\n"


usefulpath = 'E:/DeepLearning/spider/网络设备/交换机'
filelist = os.listdir(usefulpath)
o = open("test.xml", "w+")
o.writelines(getfilelist(usefulpath))
o.close()
print("成功！请查看test.xml文件" )