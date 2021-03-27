from itertools import count
from operator import index
from time import time
from tensorflow.python.ops.variables import model_variables
from IPython.display import display, Image
from keras_applications.resnet50 import ResNet50
from keras_applications.vgg16 import VGG16
from tensorflow import keras
import os
import sys
import random
import glob
import pathlib
import tensorflow as tf
import numpy as np
import cv2
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#-------------------------------------------------------------
#Author:AnalogKnight
#Time:2021.03.04
#-------------------------------------------------------------

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def PreprocessImage(image):
    try:
        image = tf.image.decode_image(image,channels=3)
        image = tf.image.resize(image, [256, 256])
        image /= 255.0  # normalize to [0,1] range
    except:
        print("Processing error!")
    return image

def LoadAndPreprocessImage(path):
    image = tf.io.read_file(path)
    return PreprocessImage(image)

data=[]

rootPath=os.path.dirname(sys.argv[0])
allImagePaths = []
labels = []
labelIndex=[]
#def Recognise():


def GetImageInfo():
    global labelIndex
    for dir in os.listdir(rootPath+'/train'):
        labels.append(dir)
        for file in os.listdir(rootPath+'/train/'+dir):
            allImagePaths.append(rootPath+'/train/'+dir+'/'+file)
    random.shuffle(allImagePaths)
    labelIndex = dict((name, index) for index, name in enumerate(labels))


imageDataSet=[]
labelDataSet=[]
def LoadData():
    global imageDataSet
    global labelDataSet
    count=0
    amount=len(allImagePaths)

    for path in allImagePaths:
        imageDataSet.append(LoadAndPreprocessImage(path))
        labelDataSet.append(labelIndex[pathlib.Path(path).parent.name])
        count+=1
        print(str(count)+'/'+str(amount))

    imageDataSet=np.array(imageDataSet)
    imageDataSet=np.squeeze(imageDataSet)
    labelDataSet=np.array(labelDataSet)

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0005

import tkinter
import tkinter.messagebox
def Report(accuracy):
    if tkinter.messagebox.askquestion('Save the model?','The accuracy is:'+str(accuracy)):
        Save()

import datetime
def Save():
    if not os.path.exists(rootPath+"/saved_model/"):
        os.makedirs(rootPath+"/saved_model/")
    model.save(rootPath+"/saved_model/"+str(datetime.datetime.now())+".h5")

def Train():
    global model
    model = VGG16(include_top=False, weights=None, input_shape=(256, 256, 3), pooling='avg', classes=2,backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    x = model.get_layer(index=18).output
# x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(2, activation='softmax', name='predictions')(x)  
    model = keras.models.Model(model.input, x)
    change_lr = keras.callbacks.LearningRateScheduler(scheduler)
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45, zoom_range=0.5, horizontal_flip=True, width_shift_range=0.15, height_shift_range=0.15
    )
    train_generator.fit(imageDataSet)
    test_generator = keras.preprocessing.image.ImageDataGenerator()
    test_generator.fit(testimageDataSet)

    model.fit_generator(
    generator=train_generator.flow(imageDataSet, labelDataSet, batch_size=32),
    epochs=150, verbose=2, callbacks=[change_lr],
    validation_data=test_generator.flow(testimageDataSet, testlabelDataSet, batch_size=32))

    test_loss, test_acc = model.evaluate(testimageDataSet,  testlabelDataSet, verbose=2)
    Report(test_acc)
    print('\nTest accuracy:', test_acc)

import matplotlib.pyplot as plt
def PlotImage(i, predictionsArray, img):
    predictionsArray, img = predictionsArray, img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictionsArray)
    '''if predicted_label == trueLabel:
        color = 'blue'
    else:
        color = 'red'
    '''

    plt.xlabel("{} {:2.0f}%".format(labels[predicted_label],100*np.max(predictionsArray)))

def PlotValueArray(i, predictionsArray):
    predictionsArray = predictionsArray
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictionsArray, color="#777777")
    plt.ylim([0, 1])
    predictedLabel = np.argmax(predictionsArray)

    thisplot[predictedLabel].set_color('blue')


def ReturnResult(predictions,targetImages,count):
    cols = 6
    if count/cols<=1:
        rows = 1
    else:
        rows=int(count/cols)+1
    #num_images = rows*cols
    plt.figure(figsize=(2*2*cols, 2*rows))
    for i in range(count):
        plt.subplot(rows, 2*cols, 2*i+1)
        PlotImage(i, predictions[i],  targetImages)
        plt.subplot(rows, 2*cols, 2*i+2)
        PlotValueArray(i, predictions[i])
    plt.tight_layout()
    plt.show()

def ProcessData():
    for dir in os.listdir(rootPath+'/raw'):
        if os.path.exists(rootPath+'/train/'+dir)==False:
            os.mkdir(rootPath+'/train/'+dir)
        for file in os.listdir(rootPath+'/raw/'+dir):
            #image = cv2.cvtColor(cv2.imread(rootPath+'/raw/'+dir +'/'+file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            image = cv2.imread(rootPath+'/raw/'+dir +'/'+file, cv2.IMREAD_COLOR)
            image = cv2.resize(image,(256,256))
            flips = []
            flips.append(cv2.flip(image, -1))
            flips.append(cv2.flip(image, 0))
            flips.append(cv2.flip(image, 1))
            print(rootPath+'/raw/'+dir +'/'+file)
            cv2.imwrite(rootPath+'/train/'+dir+'/'+file[:(len(file)-4)]+".jpg", image,[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
            for i in range(0,3):
                cv2.imwrite(rootPath+'/train/'+dir+'/'+file[:(len(file)-4)]+str(i)+".jpg", flips[i],[int(cv2.IMWRITE_PNG_COMPRESSION), 5])

targetPaths=[]
def GetTarget(path,count,cut):
    xCount=4
    targetDataSet=[]
    global targetPaths
    from PIL import Image
    for file in os.listdir(path):
        if cut==True:
            image = tf.io.read_file(path+'/'+file)
            image = tf.image.decode_image(image,channels=3)
            #image=LoadAndPreprocessImage(path+'/'+file)
            x=image.shape[1]
            y=image.shape[0]
            Size=int(x/xCount)
            yCount=int(y/Size)
            for i in range(xCount):
                for i2 in range(yCount):
                    #print(image[i*xCount:(i+1)*xCount][i2*xCount:(i2+1)*xCount])
                    piece=image[i2*Size:(i2+1)*Size,i*Size:(i+1)*Size]
                    saveFile=Image.fromarray(np.array(piece))
                    saveFile.save(rootPath+'/cut/'+file[:(len(file)-4)]+str(i)+'_'+str(i2)+".png")
                    #tf.io.gfile.GFile(rootPath+'/cut/'+file[:(len(file)-4)]+str(i)+'_'+str(i2)+".jpg",piece)
                    #cv2.imwrite(rootPath+'/cut/'+file[:(len(file)-4)]+str(i)+'_'+str(i2)+".jpg",piece,[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                    piece = tf.image.resize(piece, [256, 256])
                    piece/=255.0
                    targetDataSet.append(piece)
                    targetPaths.append(path+'/'+file)
        else:
            targetDataSet.append(LoadAndPreprocessImage(path+'/'+file))
            targetPaths.append(path+'/'+file)
        count[0]+=1
    targetDataSet=np.array(targetDataSet)
    print(targetPaths)
    return targetDataSet

def Recognize(load,path,match,detection):
    if load ==True:
        files=[]
        for saveFile in os.listdir(rootPath +"/saved_model"):
            files.append(saveFile)
        files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(rootPath +"/saved_model", x)))
        print("Loading:"+files[-1])
        global model
        model=tf.keras.models.load_model(rootPath +"/saved_model/"+files[-1])

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    count=[0]
    targetImages=GetTarget(path,count,detection)
    predictions = probability_model.predict(targetImages)
    if match!=None:
        for i in range(count[0]):
            if np.argmax(predictions[i])==labelIndex[tag]:
                print(targetPaths[i])
        return
    ReturnResult(predictions,targetImages,count[0])

'''def Detection(load,path,match):
    if load ==True:
        files=[]
        for saveFile in os.listdir(rootPath +"/saved_model"):
            files.append(saveFile)
        files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(rootPath +"/saved_model", x)))
        print("Loading:"+files[-1])
        global model
        model=tf.keras.models.load_model(rootPath +"/saved_model/"+files[-1])'''

testallImagePaths = []
testimageDataSet=[]
testlabelDataSet=[]
def GetTestData():
    #global testallImagePaths
    '''for dir in os.listdir(rootPath+'/test'):
        for file in os.listdir(rootPath+'/test/'+dir):
            testallImagePaths.append(rootPath+'/test/'+dir+'/'+file)

    random.shuffle(testallImagePaths)

    count=0
    amount=len(testallImagePaths)

    for path in testallImagePaths:
        testimageDataSet.append(LoadAndPreprocessImage(path))
        testlabelDataSet.append(labelIndex[pathlib.Path(path).parent.name])
        count+=1
        print(str(count)+'/'+str(amount))

    testimageDataSet=np.array(testimageDataSet)
    testimageDataSet=np.squeeze(testimageDataSet)
    testlabelDataSet=np.array(testlabelDataSet)'''
    global testimageDataSet
    global testlabelDataSet
    global imageDataSet
    global labelDataSet
    count=len(imageDataSet)
    position=count-int(count/5)
    testimageDataSet=imageDataSet[position:]
    testlabelDataSet=labelDataSet[position:]
    imageDataSet=imageDataSet[:position]
    labelDataSet=labelDataSet[:position]

if len(sys.argv)>1:
    operation=int(sys.argv[1])
    if operation==0:
        ProcessData()
    elif operation==1:
        #data=sys.argv[2].split(',')
        GetImageInfo()
        #LoadData()
        #GetTestData()
        Recognize(True,sys.argv[2],None,False)
    elif operation==2:
        #data=sys.argv[2].split(',')
        GetImageInfo()
        LoadData()
        GetTestData()
        Train()
    elif operation==3:
        GetImageInfo()
        existent=False
        tag=''
        if len(sys.argv)<=3:
            print("Missing tag.")
        else:
            tag=sys.argv[3]
        for item in labels:
            if tag==item:
                existent=True
        if existent:
            Recognize(True,sys.argv[2],tag,False)
        else:
            print("Tag not existent.")
