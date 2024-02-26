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

fileNumber = 0

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

std = 21
swing = 2

resultMsg = ""
resultPath = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\output.txt".format(std=std, swing=swing)


#生成這個frame的向量資訊(讀取json)
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

    return myPosition

 #生成這個frame的向量資訊(讀取json)
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

#生成merge出來的的向量資訊(讀取list)
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

#回傳兩個向量的角度差
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

#因為圖片的原點在左上角，所以需要交換
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

path_to_Target = 'sample_data/'
# Target_json_files = [pos_json for pos_json in os.listdir(path_to_Target) if pos_json.endswith('.json')]
# for i in range(len(Target_json_files)):
    # sampleSwingVector.append(generateVector(path_to_Target+Target_json_files[i]))

keyPointVector = []
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing1_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing2_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing3_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing4_keypoints.json'))
keyPointVector.append(generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/keyAction_Key/swing5_keypoints.json'))
#keyPointVector.append(generateVector('keyAction_Key/swing1_keypoints.json'))


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

shutil.rmtree('swingList', ignore_errors=True)
ready_begin_compareVector = generateVector('C:/Users/kelly/Desktop/NCHU-NLPLab-PEGolf/OpenPose/Golf_Run/realTimeJudge/SampleReadyKey/sample_keypoints.json') # 取得要比對的向量資訊
#ready_begin_compareVector = generateVector('SampleReadyKey/sample_keypoints.json') # 取得要比對的向量資訊

# lastFiles = glob.glob('Mykey/*')
# for f in lastFiles:
    # os.remove(f)
subprocess.Popen(["C:\\Users\\kelly\\Desktop\\NCHU-NLPLab-PEGolf\\OpenPose\\Golf_Run\\run\\OpenPoseDemo.exe","--model_folder","C:\\Users\\kelly\\Desktop\\NCHU-NLPLab-PEGolf\\OpenPose\\Golf_Run\\models","--camera","0","--num_gpu_start","0","--process_real_time","-number_people_max","1","--net_resolution","480x368","--write_json","D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\".format(std=std, swing=swing)])
time.sleep(1)


while True:
    nextFileNumber = fileNumber+1
    numberDigits_1 = len(str(fileNumber))
    numberDigits_2 = len(str(nextFileNumber))
    fileName = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\Student_1_{std}_{swing}_".format(std=std, swing=swing)
    nextFileName = "D:\\DataSet\\202306\\0601_1\\Student_1_{std}\\output\\json\\swing{swing}\\Student_1_{std}_{swing}_".format(std=std, swing=swing)
    #fileName = 'StudentData/json/Student_1_{std}_{swing}_'.format(std=std, swing=swing)
    #nextFileName = 'StudentData/json/Student_1_{std}_{swing}_'.format(std=std, swing=swing)
    for i in range(0,12-numberDigits_1):
        fileName += '0'
    for i in range(0,12-numberDigits_2):
        nextFileName += '0'
    fileName += str(fileNumber)
    fileName += '_keypoints.json'
    nextFileName += str(nextFileNumber)
    nextFileName += '_keypoints.json'
    print(fileName)

    while not os.path.exists(nextFileName): #下一個檔案存在才會去進行
        time.sleep(0.01)

    #successfully read current frame
    with open(fileName) as f:
        data = json.load(f)
        if len(data["people"]) > 0 :
            currentVector = generateVector(fileName) #取得target的向量資訊
            currentPos = generatePosition(fileName) #取得target的向量資訊

            readyScore = VectorDiffLoss(currentVector,ready_begin_compareVector)
            
            if readyScore <= READY_SCORE_THRESHOLD and angleDiff(ready_begin_compareVector[9],currentVector[9]) < 15: #如果符合預備動作
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
                print("Ready")
                print(fileName)

                '''
                if readyCount == 15: #如果有重新檢測到15 Frames的Ready動作，代表重新進行揮桿，將資料清除
                    swingState = 1
                    leftLift = False
                    rightLift = False
                    peakAngle = 0
                    dropCount = 0
                    wholeSwingVector.clear()
                    wholeSwingPosition.clear()
                    readyArmVector = currentVector[3] #用上手臂來計算角度
                    wholeSwingVector.append(currentVector)
                    wholeSwingPosition.append(currentPos)
                    print("ready")
                '''
            else: #只要不符合就會清0
                readyCount = 0

            if swingState == 1: #已經檢測到準備完成，但可能還在準備動作
                if readyScore > READY_SCORE_THRESHOLD: #如果不是預備動作 (代表開始揮桿了)
                    resultMsg += "Ready"
                    resultMsg += ","
                    resultMsg += fileName
                    resultMsg += "\n"

                    with open(resultPath, 'a') as f:
                        f.write(resultMsg)

                    print("Swing Start")
                    print(fileName)
                    swingState = 2
            elif swingState == 2: #揮桿動作 (leftSwing)
                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                if(angle_counterClockwise(readyArmVector, currentVector[3]) > 180 and angle_clockwise(readyArmVector, currentVector[3]) >= 25) : #leftSwing完成 , 與上手臂進行比對
                    resultMsg += "Left ok"
                    resultMsg += ","
                    resultMsg += fileName
                    resultMsg += "\n"

                    with open(resultPath, 'a') as f:
                        f.write(resultMsg)

                    print("Left ok")
                    print(fileName)
                    swingState = 3
            elif swingState == 3: #節錄動作 (rightSwing)
                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                #print(angle_counterClockwise(readyArmVector, currentVector[3]))
                if(angle_clockwise(readyArmVector, currentVector[3]) > 180 and angle_counterClockwise(readyArmVector, currentVector[3]) >= 18) : #rightSwing完成
                    resultMsg += "Right ok"
                    resultMsg += ","
                    resultMsg += fileName
                    resultMsg += "\n"

                    with open(resultPath, 'a') as f:
                        f.write(resultMsg)

                    print("Right ok")
                    print(fileName)
                    swingState = 4    
            elif swingState == 4: #節錄動作 (收桿)
                wholeSwingVector.append(currentVector)
                wholeSwingPosition.append(currentPos)
                currentAngle = angle_counterClockwise(readyArmVector, currentVector[3])

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
                        resultMsg += "Swing Finish"
                        resultMsg += ","
                        resultMsg += fileName
                        resultMsg += "\n"

                        with open(resultPath, 'a') as f:
                            f.write(resultMsg)

                        print("Swing Finish")
                        print(fileName)

                        # 註解
                        # 繪製揮桿的每一幀動作
                        '''
                        Path("swingList/output_" + str(swingCount)).mkdir(parents=True, exist_ok=True)
                        for i in range(len(wholeSwingVector)) :
                            #繪製比對動作的骨架
                            plt.rcParams["figure.figsize"] = [4, 6]
                            plt.rcParams["figure.autolayout"] = True
                            plt.figure(figsize=(4, 6), dpi = 100)
                            soa = np.array([0,0,0,0])

                            for j in range(16) :
                                newrow = [wholeSwingPosition[i][j][0],wholeSwingPosition[i][j][1],wholeSwingVector[i][j][0],wholeSwingVector[i][j][1]]
                                soa = np.vstack([soa, newrow])

                            X, Y, U, V = zip(*soa)
                            ax = plt.gca()
                            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
                            ax.set_xlim([0, 800])
                            ax.set_ylim([300, 1850])
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            for j in range(16) :
                                plt.scatter(wholeSwingPosition[i][j][0], wholeSwingPosition[i][j][1], s=50)

                            plt.gca().invert_yaxis()
                            output_fileName = "swingList/output_" + str(swingCount) + "/" + str(i)
                            plt.savefig(output_fileName+".png")
                            plt.clf()
                            plt.close('all')
                        #
                        '''


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
                        '''
                        Path("swingList/output_" + str(swingCount)).mkdir(parents=True, exist_ok=True)
                        for i in range(5) :
                            #繪製比對動作的骨架
                            plt.rcParams["figure.figsize"] = [4, 6]
                            plt.rcParams["figure.autolayout"] = True
                            plt.figure(figsize=(4, 6), dpi = 100)
                            soa = np.array([0,0,0,0])

                            for j in range(16) :
                                newrow = [wholeSwingPosition[KeyActionList[i]][j][0],wholeSwingPosition[KeyActionList[i]][j][1],wholeSwingVector[KeyActionList[i]][j][0],wholeSwingVector[KeyActionList[i]][j][1]]
                                soa = np.vstack([soa, newrow])

                            X, Y, U, V = zip(*soa)
                            ax = plt.gca()
                            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
                            ax.set_xlim([0, 500])
                            ax.set_ylim([0, 880])
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            for j in range(16) :
                                plt.scatter(wholeSwingPosition[KeyActionList[i]][j][0], wholeSwingPosition[KeyActionList[i]][j][1], s=50)

                            plt.gca().invert_yaxis()
                            output_fileName = "swingList/output_" + str(swingCount) + "/" + str(i)
                            plt.savefig(output_fileName+".png")
                            plt.clf()
                            plt.close('all')
                        #
                        '''

                        '''
                        這邊可以sleep大概1秒鐘，高爾夫球擊球路徑的分數會寫在某個檔案裡面，等待一秒鐘之後去讀取該檔案
                        之後可以參考sendState來把訊息傳送到前端App上
                        '''


                        resultMsg += "FinalScore:{}".format(finalScore)
                        resultMsg += "\n"
                        print("FinalScore:{}".format(finalScore))
                        print(resultMsg)

                        with open(resultPath, 'a') as f:
                            f.write(resultMsg)

                        swingCount += 1
                        swingState = 0
    
    #os.remove(fileName)
    fileNumber += 1

