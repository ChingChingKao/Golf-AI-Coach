import sys
from http.client import OK
from pickle import TRUE
from pickletools import read_bytes1
import subprocess
from tkinter import END
import numpy as np
import os, json
import pandas as pd
import math
from math import acos
from math import sqrt
from math import pi
import time
import glob
import matplotlib.pyplot as plt
import socket
import cv2
from pathlib import Path
import shutil

READY_SCORE_THRESHOLD = 30

fileNumber = 2

readyCount = 0
swingState = 0
wholeSwingVector = []
wholeSwingPosition = []
leftLift = False
rightLift = False

readyArmVector = None
dropCount = 0
swingCount = 0
score = 0

peakNum = 0

resultMsg = ""

# 生成這個frame的向量資訊(讀取json)
def generatePosition(fileName):
    myPosition = []
    with open(fileName) as f:
        data = json.load(f)
        keys = data["people"][0]["pose_keypoints_2d"]

        #head
        myPosition.append([keys[0],keys[1]])

        #body
        myPosition.append([keys[3],keys[4]])
        
        #right hand
        myPosition.append([keys[3],keys[4]])
        myPosition.append([keys[6],keys[7]])
        myPosition.append([keys[9],keys[10]])

        
        #left hand
        myPosition.append([keys[3],keys[4]])
        myPosition.append([keys[15],keys[16]])
        myPosition.append([keys[18],keys[19]])
    
        
        #right leg
        myPosition.append([keys[24],keys[25]])
        myPosition.append([keys[27],keys[28]])
        myPosition.append([keys[30],keys[31]])
        myPosition.append([keys[33],keys[34]])
       
        
        #left leg
        myPosition.append([keys[24],keys[25]])
        myPosition.append([keys[36],keys[37]])
        myPosition.append([keys[39],keys[40]])
        myPosition.append([keys[42],keys[43]])

        myPosition.append([keys[12],keys[13]])
        myPosition.append([keys[21],keys[22]])


    return myPosition

# 生成這個frame的向量資訊(讀取json)
def generateVector(fileName):
    myVector = []
    with open(fileName) as f:
        data = json.load(f)
        keys = data["people"][0]["pose_keypoints_2d"]

        #head
        if keys[2] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[3]-keys[0],keys[4]-keys[1]])

        #body
        if keys[26] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[24]-keys[3],keys[25]-keys[4]])
        
        #right hand
        if keys[8] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else :
            myVector.append([keys[6]-keys[3],keys[7]-keys[4]])

        if keys[11] == 0 or keys[8] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[9]-keys[6],keys[10]-keys[7]])
       
        if keys[14] == 0 or keys[11] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[12]-keys[9],keys[13]-keys[10]])
        
        #left hand
        if keys[17] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[15]-keys[3],keys[16]-keys[4]])
        
        if keys[20] == 0 or keys[17] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[18]-keys[15],keys[19]-keys[16]])
        
        if keys[23] == 0 or keys[20] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[21]-keys[18],keys[22]-keys[19]])
        
        #right leg
        if keys[29] == 0 or keys[26] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[27]-keys[24],keys[28]-keys[25]])

        if keys[32] == 0 or keys[29] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[30]-keys[27],keys[31]-keys[28]])
        
        if keys[35] == 0 or keys[32] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[33]-keys[30],keys[34]-keys[31]])
        
        if keys[68] == 0 or keys[35] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[66]-keys[33],keys[67]-keys[34]])
        
        #left leg
        if keys[38] == 0 or keys[26] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[36]-keys[24],keys[37]-keys[25]])
        
        if keys[41] == 0 or keys[38] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[39]-keys[36],keys[40]-keys[37]])
        
        if keys[44] == 0 or keys[41] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[42]-keys[39],keys[43]-keys[40]])
        
        if keys[59] == 0 or keys[44] == 0 : #若有一個點沒偵測到,向量設為0
            myVector.append([0,0])
        else:
            myVector.append([keys[57]-keys[42],keys[58]-keys[43]])
        
    return myVector

# 生成merge出來的的向量資訊(讀取list)
def generateVectorFromList(keys):
    myVector = []
    #head
    if keys[2] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[3]-keys[0],keys[4]-keys[1]])

    #body
    if keys[26] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[24]-keys[3],keys[25]-keys[4]])
    
    #right hand
    if keys[8] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else :
        myVector.append([keys[6]-keys[3],keys[7]-keys[4]])

    if keys[11] == 0 or keys[8] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[9]-keys[6],keys[10]-keys[7]])
    
    if keys[14] == 0 or keys[11] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[12]-keys[9],keys[13]-keys[10]])
    
    #left hand
    if keys[17] == 0 or keys[5] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[15]-keys[3],keys[16]-keys[4]])
    
    if keys[20] == 0 or keys[17] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[18]-keys[15],keys[19]-keys[16]])
    
    if keys[23] == 0 or keys[20] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[21]-keys[18],keys[22]-keys[19]])
    
    #right leg
    if keys[29] == 0 or keys[26] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[27]-keys[24],keys[28]-keys[25]])

    if keys[32] == 0 or keys[29] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[30]-keys[27],keys[31]-keys[28]])
    
    if keys[35] == 0 or keys[32] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[33]-keys[30],keys[34]-keys[31]])
    
    if keys[68] == 0 or keys[35] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[66]-keys[33],keys[67]-keys[34]])
    
    #left leg
    if keys[38] == 0 or keys[26] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[36]-keys[24],keys[37]-keys[25]])
    
    if keys[41] == 0 or keys[38] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[39]-keys[36],keys[40]-keys[37]])
    
    if keys[44] == 0 or keys[41] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[42]-keys[39],keys[43]-keys[40]])
    
    if keys[59] == 0 or keys[44] == 0 : #若有一個點沒偵測到,向量設為0
        myVector.append([0,0])
    else:
        myVector.append([keys[57]-keys[42],keys[58]-keys[43]])
        
    return myVector

# 回傳兩個向量的角度差
def angleDiff(vector_1,vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if dot_product > 1 :
        dot_product = 1
    angle = np.arccos(dot_product)
    return angle*57.3

def mergeFrame(number,VectorFrame_1,VectorFrame_2) :
    file_2_percent = number-math.floor(number)
    file_1_percent = 1-file_2_percent
    mergeList = []
    i = 0
    while i < 16 :
        mergeList.append([VectorFrame_1[i][0]*file_1_percent,VectorFrame_1[i][1]*file_1_percent])
        i += 1

    while i < 16 :
        if mergeList[i][0] == 0 and  mergeList[i][1] == 0:#如果上一個frame沒偵測到,那就直接使用這個frame的資訊
            mergeList[i] = [VectorFrame_2[i][0],VectorFrame_2[i][1]]
        else:
            mergeList[i][0] += VectorFrame_2[i][0]*file_2_percent
            mergeList[i][1] += VectorFrame_2[i][1]*file_2_percent
        i += 1

    return mergeList

def VectorDiffLoss(vec1,vec2):
    points = 0
    i = 2
    zeroVectorCount = 0
    while i <= 7 :
        if(vec1[i][0] == 0 and vec1[i][1] == 0) or (vec2[i][0] == 0 and vec2[i][1] == 0) :#如果有一個是0向量
            zeroVectorCount += 1
        else :
            t = angleDiff(vec1[i],vec2[i])
            t = (t/6.25)**(4) #相差5度內沒什麼關係
            points += t
        i+=1
    if zeroVectorCount >= 6: #如果超過8個向量沒被偵測,讓分數一次加很多
        return 500000
    return points/(6-zeroVectorCount)

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def angle_counterClockwise(A, B): #因為圖片的原點在左上角，所以需要交換
    if A[0] == 0 or A[1] == 0 or B[0] == 0 or B[1] == 0:
        return -1
    if(abs(A[0]-B[0])+abs(A[1]-B[1])) < 1:
        return 0
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

# 因為圖片的原點在左上角，所以需要交換
def angle_clockwise(A, B):
    if A[0] == 0 or A[1] == 0 or B[0] == 0 or B[1] == 0:
        return -1
    if(abs(A[0]-B[0])+abs(A[1]-B[1])) < 1:
        return 0
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return 360-inner
    else: # if the det > 0 then A is immediately clockwise of B
        return inner

a = np.zeros(shape=70)
haveKeyPoint = np.zeros(shape=5)
sampleSwingVector = []

#path_to_Target = 'sample_data/'
#Target_json_files = [pos_json for pos_json in os.listdir(path_to_Target) if pos_json.endswith('.json')]
#for i in range(len(Target_json_files)):
    #sampleSwingVector.append(generateVector(path_to_Target+Target_json_files[i]))

keyPointVector = []
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing1_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing2_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing3_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing4_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing5_keypoints.json'))


def scoreCalculate(myVector, current, total):
    minLoss = 999999999
    targetNumber = (float(current)/float(total)) * 70
    upperBound = 0
    lowerBound = 0
    if targetNumber-5 < 0:
        lowerBound = 0
    else:
        lowerBound = int(targetNumber)-5

    if targetNumber+5 >= 70:
        upperBound = 69
    else:
        upperBound = int(targetNumber)+5
    for i in range(lowerBound, upperBound+1):
        lossValue = VectorDiffLoss(myVector,sampleSwingVector[i])
        if lossValue < minLoss:
            minLoss = lossValue
    return minLoss

def chooseTheBest(myVector):
    minLoss = 999999999
    chosen = 0
    for i in range(len(sampleSwingVector)):
        lossValue = VectorDiffLoss(myVector,sampleSwingVector[i])
        if lossValue < minLoss:
            chosen = i
            minLoss = lossValue
    a[chosen] = 1

def detectKeypoint(myVector,num,KeyActionList):
    for i in range(5):
        lossValue = VectorDiffLoss(myVector,keyPointVector[i])
        if lossValue < haveKeyPoint[i]:
            haveKeyPoint[i] = lossValue
            KeyActionList[i] = num
    
fileNumber = 0
std = 19
swing = 3
dirpath = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\".format(std=std, swing=swing)
#dirpath = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\swing{swing}\\".format(std=std, swing=swing)
ready_begin_compareVector = generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/SampleReadyKey/sample_keypoints.json') # 取得要比對的向量資訊
print("1")
status = [0,0,0,0,0]
keyP = []
result =[]
swingState = 0
for iii in range(len(os.listdir(dirpath))-1):
    nextFileNumber = fileNumber+1
    numberDigits_1 = len(str(fileNumber))
    numberDigits_2 = len(str(nextFileNumber))
    fileName = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\Student_1_{std}_{swing}_".format(std=std, swing=swing)
    #fileName = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\swing{swing}\\720164978.196114_".format(std=std, swing=swing)
    nextFileName = fileName

    #'D:\\users\\andylin77\\Documents\\NCHU\\ProjectFile\\PE_Golf\\DataSet\\202306\\0601_1\\Student_1_1\\output\\json\\swing2\\Student_1_1_2_'
    for i in range(0,12-numberDigits_1):
        fileName += '0'
    for i in range(0,12-numberDigits_2):
        nextFileName += '0'
    fileName += str(fileNumber)
    fileName += '_keypoints.json'
    nextFileName += str(nextFileNumber)
    nextFileName += '_keypoints.json'
    
    print("2 "+ str(iii))
    with open(fileName) as f:
        debug = fileName
        data = json.load(f)
        if len(data["people"]) > 0 :
            currentVector = generateVector(fileName) #取得target的向量資訊
            currentPos = generatePosition(fileName) #取得target的向量資訊

            readyScore = VectorDiffLoss(currentVector,ready_begin_compareVector)
            
            if swingState <=1 and readyScore <= READY_SCORE_THRESHOLD and angleDiff(ready_begin_compareVector[9],currentVector[9]) < 15: #如果符合預備動作
                readyCount += 1
                swingState = 1
                leftLift = False
                rightLift = False
                peakAngle = 0
                dropCount = 0
                wholeSwingVector.clear()
                wholeSwingPosition.clear()
                readyArmVector = currentVector[3]  # 用上手臂來計算角度
                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                readyPoseRecord = fileName
                print("Ready")

            else: #只要不符合就會清0
                readyCount = 0

            if swingState == 1: #已經檢測到準備完成，但可能還在準備動作
                if readyScore > READY_SCORE_THRESHOLD: #如果不是預備動作 (代表開始揮桿了)
                    print("Swing Start")
                    #print(fileName)
                    swingState = 2

            elif swingState == 2: #上桿動作 (leftSwing)
                if status[2] == 0:
                    #準備結束 開始揮桿
                    status[2] = 1
        
                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                if(angle_counterClockwise(wholeSwingVector[-2][3], currentVector[3]) < 180 and abs(180-angle_counterClockwise([10,0.01], currentVector[3])) < 30) : #leftSwing完成 , 與上手臂進行比對
                    print("Left ok")
                    swingState = 3
            elif swingState == 3: #下桿動作 (rightSwing)
                if(status[3] == 0):
                    #上桿結束 上桿判斷
                    #身體為旋轉2,5
                    print(fileName)
                    if(angle_counterClockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) > 65 and angle_counterClockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) < 110 ):#身體未旋轉
                        result.append('上桿身體未旋轉')
                    if(angle_clockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) > 65 and angle_clockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) < 110 ):#身體未旋轉
                        result.append('上桿身體未旋轉')
                    #手臂彎曲
                    if(angle_counterClockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) < 60 or angle_counterClockwise([currentPos[6][0]-currentPos[7][0],currentPos[6][1]-currentPos[7][1]], [currentPos[17][0]-currentPos[7][0],currentPos[17][1]-currentPos[7][1]]) < 135):
                        result.append('上桿手臂彎曲太多')

                    #手臂三角
                    #or angle_counterClockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) < 30 or angle_counterClockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) > 65
                    if(angle_counterClockwise([currentPos[6][0]-currentPos[7][0],currentPos[6][1]-currentPos[7][1]], [currentPos[17][0]-currentPos[7][0],currentPos[17][1]-currentPos[7][1]]) < 160 ):
                        result.append('上桿手臂三角不穩定')
                    

                    status[3] = 1

                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)

                #print(angle_counterClockwise(readyArmVector, currentVector[3]))
                if(angle_clockwise(readyArmVector, currentVector[3]) < 10 or angle_clockwise(readyArmVector, currentVector[3]) >350) : #rightSwing完成
                    print("Right ok")
                    keyP=wholeSwingPosition[-3:]
                    swingState = 4 
                    print(fileName)

            elif swingState == 4: #收桿動作
                if status[4] == 0:
                    #下桿結束 下桿判斷
                    status[4] = 1
                    #身體為旋轉2,5
                    if(angle_counterClockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) > 65  and angle_counterClockwise([0.01,10], [currentPos[3][0]-currentPos[6][0],currentPos[3][1]-currentPos[6][1]]) < 110 ):#身體未旋轉
                        result.append('下桿身體未旋轉')
                    
                    #手臂三角
                    #or angle_counterClockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) < 30 or angle_counterClockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) > 65
                    if(angle_counterClockwise([currentPos[6][0]-currentPos[7][0],currentPos[6][1]-currentPos[7][1]], [currentPos[17][0]-currentPos[7][0],currentPos[17][1]-currentPos[7][1]]) < 160 ):
                        result.append('下桿手臂三角不穩定放鬆擊球')

                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                currentAngle = angle_counterClockwise(readyArmVector, currentVector[3])
    
                if currentAngle < 50:
                    if (status[4] ==1 and angle_clockwise([currentPos[3][0]-currentPos[4][0],currentPos[3][1]-currentPos[4][1]], [currentPos[16][0]-currentPos[4][0],currentPos[16][1]-currentPos[4][1]]) < 160):
                        result.append('送桿延伸未到位')
                        status[4] = 2 

                if currentAngle > 250:
                    currentAngle = 360-currentAngle

                if(currentAngle < 250 and currentAngle > peakAngle) :
                    peakNum = len(wholeSwingVector)-1
                    peakAngle = currentAngle
                    dropCount = 0
                elif currentAngle != -1: # and peakAngle - currentAngle >= 20:
                    dropCount += 1
                    if dropCount == 3:
                        #結束揮桿
                        #收桿 不完整

                        print(fileName)

                        if(status[4] == 2 or angle_counterClockwise([-10,0.01], [currentPos[1][0]-currentPos[17][0],currentPos[1][1]-currentPos[17][1]]) < 75 ):#身體未旋轉
                            result.append('收桿不完整')
                        print("Swing Finish")
                        
                        # 註解
                        # 繪製揮桿的每一幀動作
                       
                        KeyActionList = []
                        for i in range(5):
                            KeyActionList.append(-1)
                        
                        haveKeyPoint = np.zeros(shape=5)
                        haveKeyPoint[0] = 1000000
                        haveKeyPoint[1] = 1000000
                        haveKeyPoint[2] = 1000000
                        haveKeyPoint[3] = 1000000
                        haveKeyPoint[4] = 1000000

                        for i in range(len(wholeSwingVector)-3-(int(len(wholeSwingVector)*0.1))):
                            detectKeypoint(wholeSwingVector[i],i,KeyActionList)

                        finalScore = 0

                        if haveKeyPoint[0] <= 100:
                            finalScore += 1
                        if haveKeyPoint[1] <= 150:
                            finalScore += 1

                        if haveKeyPoint[2] <= 500:
                            finalScore += 1
                        elif haveKeyPoint[2] <= 1000:
                            finalScore += 0.5
                        
                        if haveKeyPoint[3] <= 1000:
                            finalScore += 1
                        elif haveKeyPoint[3] <= 1800:
                            finalScore += 0.5
                        
                        if haveKeyPoint[4] <= 500:
                            finalScore += 1
                        elif haveKeyPoint[4] <= 1800:
                            finalScore += 0.5
                        
                        # 註解
                        # 繪製與要比較的5個動作之最相似動作
                        
                        resultMsg += "FinalScore:{}".format(finalScore)
                        resultMsg += "\n"
                        #print("FinalScore:{}".format(finalScore))
                        print(resultMsg)
                        swingCount += 1
                        swingState = 6
                        break

    
    fileNumber += 1
outfile = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\result.json".format(std=std, swing=swing)
#outfile = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\swing{swing}\\result.json".format(std=std, swing=swing)
if swingState == 0:
    result.append('影片中沒有揮桿動作')
with open(outfile,'w+',encoding='utf-8')as file:  
    json.dump(result,file,ensure_ascii=False) 
