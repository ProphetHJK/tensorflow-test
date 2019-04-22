#coding:utf-8
import os
import os.path
from PIL import Image
import shutil
path = "D:\\picture\\Users\\12567\\Pictures"
files = os.listdir(path)
s=[]
def Traversing(rootDir):
    """遍历文件
    arg1 : string
        目录
    """
    list_dirs=os.walk(rootDir)
    for root,dirs,files in list_dirs:
        for f in files:
            if os.path.splitext(f)[-1] == ".jpg":
                print(f)
                frompath = root
                im = Image.open(frompath+"\\"+f)
                imsize = im.size[0]*im.size[1]#计算像素
                topath = frompath.replace("D:\\", "D:\\picture\\")
                if imsize > 16000000:
                    if os.path.exists(topath) == False:
                        os.makedirs(topath)#如目录不存在则创建目录
                    shutil.copyfile(frompath+"\\"+f, topath+"\\"+f)


def ShowTraversing(rootDir):
    """打印的遍历文件到文件
    arg1 : string
        目录
    """
    list_dirs = os.walk(rootDir)
    printfile = open("D:/picturelist.txt", "w+",  encoding='utf-8', errors='ignore')
    for root, dirs, files in list_dirs:
        for f in files:
            if os.path.splitext(f)[-1] == ".jpg":
                print(root+"\\"+f,file = printfile)

ShowTraversing(path)


