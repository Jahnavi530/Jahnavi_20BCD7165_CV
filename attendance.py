import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path= 'aimages'
imgs=[]
names=[]
list= os.listdir(path)
for i in list:
    cimg= cv2.imread(f'{path}/{i}')
    imgs.append(cimg)
    names.append(os.path.splitext(i)[0])
# print(names)

def finde(imgs):
    elist=[]
    for i in imgs:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        ecode= face_recognition.face_encodings(i)[0]
        elist.append(ecode)
    return elist

def markA(name):
    with open('attendance.csv','r+') as f:
        mydlist= f.readlines()
        namel=[]
        for l in mydlist:
            e= l.split(',')
            namel.append(e[0])
        if name not in namel:
            now= datetime.now()
            dts= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dts}')

elistkf= finde(imgs)
# print('encode done')

cap= cv2.VideoCapture(0)

while True:
    s,img= cap.read()
    imgsm= cv2.resize(img,(0,0),None, 0.25,0.25)
    imgsm = cv2.cvtColor(imgsm, cv2.COLOR_BGR2RGB)
    facescf = face_recognition.face_locations(imgsm)
    ecodecf = face_recognition.face_encodings(imgsm,facescf)

    for ecface, facel in zip(ecodecf,facescf):
        matches= face_recognition.compare_faces(elistkf,ecface)
        facedist= face_recognition.face_distance(elistkf,ecface)
        matchi= np.argmin(facedist)

        if matches[matchi]:
            name= names[matchi]
            y1,x2,y2,x1= facel
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markA(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

# f