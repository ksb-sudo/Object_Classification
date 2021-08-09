
#THE SPARKS FOUNDATION


#TASK-1

#OBJECT CLASSIFICATION USING OPENCV

#DONE BY-KS BALAJI


#IMPORT LIBRARIES
import cv2

#IMPORTING VIDEOS
cap=cv2.VideoCapture('people.mp4')

#########################################################
classNames=[]
#OPENING THE FILE AND STORING THE NAMES IN CLASSNAMES LIST
f=open('coco.names','r')
lines=f.readlines()

for line in lines:
    classNames=classNames+line.rstrip('\n').split('\n')

##########################################################
#INTIALIZING THE MODEL AND SETTING DEFAULT VALUES
intialization_Model='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthPath='frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weigthPath,intialization_Model)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

###########################################################


while True:
    success,img=cap.read()
    #LESSER THE THRESHOLD VALUE THAT IS CONFTHRESHOLD MORE THE DETECTION ACCURACY
    #VALUE CAN BE CHANGED DEPENDING UPON THE REQUIRMENTS
    classIds,conf,bbox=net.detect(img,confThreshold=0.5)

    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),conf.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            cv2.putText(img,str(round(confidence*100,2)), (box[0] +150, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0,255),2)


    cv2.imshow('image',img)
    cv2.waitKey(1)

############################################################
