import face_recognition
import os
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime

path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
df = pd.DataFrame(pd.read_csv("Attendance.csv"))
df = df.set_index("Name")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
images = images[1:]
classNames = classNames[1:]
def findEncodings(images):
    encodeList = []
    for img in images:
        if img is not None:
            print("Image loaded successfully.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        
    
    return encodeList
encodeListKnown = findEncodings(images)

print("Encoding Complete")
cap = cv2.VideoCapture(0)

def newFace(img):
    name = input("Enter the name of the person: ")
    new_face = img
    
    
    cv2.imwrite("ImagesAttendance/"+name+".jpg",new_face)
    print("Image saved successfully.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(encode)
    print("New photo encoded")
    classNames.append(name)
    print("Name added to database")
    #This adds the name to the csv file
    
    df.loc[name] = [datetime.now()]
    df.to_csv("Attendance.csv")

def lastSeen(name):
    lastSeen = df.loc[name].iloc[0]
    df.loc[name].iloc[0] = datetime.now().strftime("%Y-%m-%d")
    df.to_csv("Attendance.csv")
    return lastSeen

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurFrame =  face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,facesCurFrame)
    for encodeFace, faceLoc  in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex = np.argmin(faceDis)
        print(faceDis[matchIndex])


        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            date_format = "%Y-%m-%d"
            LastSeen = lastSeen(name)
            #print(type(LastSeen))
            LastSeen = LastSeen.apply(lambda x : datetime.strptime(x, date_format))
            #print((datetime.now() - LastSeen))
            cv2.putText(img,f"{name} {(datetime.now() - LastSeen)}",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            time.sleep(4)
        else:
            add_face = input("Do you want to add this face to the database? (y/n)")
            if add_face == 'y':
                newFace(img)
            


        
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)