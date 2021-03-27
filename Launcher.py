from tkinter import *
import tkinter.filedialog
import os
import time

import os
import shutil

rootPath=os.path.dirname(sys.argv[0])#获得项目根目录

def ImportData():#导入训练所需的数据
    dir=FileSelect()
    if not os.path.exists(dir):
    # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(dir)
    shutil.copytree(dir, rootPath+"/raw/"+str(os.path.split(dir)[1]))
    os.system("cd "+rootPath+"&&"+"python ./Core.py 0")

def FileSelect():#弹出系统的文件选择器，选择文件夹以输入数据
    folderDirectory = tkinter.filedialog.askdirectory()
    return folderDirectory

def Train():#调用训练功能
    os.system("cd "+rootPath+"&&"+"python ./Core.py 2")
    #print(res.read)

def Process():
    os.system("cd "+rootPath+"&&"+"python ./Core.py 0")

def Recognize():#调用图像分类功能
    dir=FileSelect()
    print(dir)
    os.system("cd "+rootPath+"&&"+"python ./Core.py 1 "+dir+'/')

#----------------------------------------------------------
#定义窗口
window= Tk()
window.title('Launcher')
window.geometry('600x400')
#----------------------------------------------------------



#----------------------------------------------------------
#创建三个主要功能按钮
importData=Button(window,text='Import',command=ImportData)
process=Button(window,text='Process data',command=Process)
train=Button(window,text='Train',command=Train)
recognize=Button(window,text='Recognize',command=Recognize)
#----------------------------------------------------------



#----------------------------------------------------------
#死循环，不断更新窗口状态
while(True):
    time.sleep(0.05)
    width=window.winfo_width()
    height=window.winfo_height()
    importData.place(x=width/2-width/6,y=height*1/5,height=height/8,width=width/3)
    process.place(x=width/2-width/6,y=height*2/5,height=height/8,width=width/3)
    train.place(x=width/2-width/6,y=height*3/5,height=height/8,width=width/3)
    recognize.place(x=width/2-width/6,y=height*4/5,height=height/8,width=width/3)
    window.update()
#----------------------------------------------------------