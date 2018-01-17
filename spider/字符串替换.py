src = '//img30.360buyimg.com/n0/s48x48_jfs/t13264/40/2323969896/144249/2ee3286d/5a3b1eb2Ne132dfb4.jpg'
#//img30.360buyimg.com/shaidan/s616x405_jfs/t13264/40/2323969896/144249/2ee3286d/5a3b1eb2Ne132dfb4.jpg
t_path = '联想（Lenovo） IBM服务器硬盘System X86专用2.5英寸含支架热插拔硬盘 146G '
if t_path[-1] == ' ':
    t_path = t_path[:-1]
    print(t_path)


'''
#去掉星号
names = '联想（Lenovo） TS250 服务器 塔式服务器 服务器主机 台式机 奔腾双核 G4560 3.5G 2*8G内存'
str = names.split(' ')
if(len(str)>=5):
    join_str = str[0]+' '+str[1]+' '+str[2]+' '+str[3]+' '+str[4]
    print(join_str)
'''

'''
tempSrc = src.split('/')
print(tempSrc)
curSrc = ''
num = 0
for i in tempSrc:
    if num == 3:
        i='shaidan'
    if num == 4:
        i='s616x405_jfs'

    curSrc+='/'
    curSrc+=i
    num+=1
curSrc = ''+curSrc[1:]
print(curSrc)
'''