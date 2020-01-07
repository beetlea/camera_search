import numpy as np
import cv2

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from imageai.Detection import ObjectDetection
import os
import serial
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel(detection_speed="flash")
sotni = int()
ser = serial.Serial('/dev/ttyACM0', 115200,  timeout=0.2)
cap = cv2.VideoCapture(1)
detect1 = int()
str1 = [0,0,0,0]
str1[0] = 0   
str1[2] = 0 

u = int()
def search_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    y2 = 0
    y1 = 0
    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        #x1 = x
        y1 = y
        #x2 = x+w
        y2 = h

        #frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
    return y2, y1

def step():
    i = 1
    u = 0
    while i:
        while u!=1:

            u = ser.write( b'v') 
        u=0 
        ff=ser.read(1)
        print(1111)
        if ff==b'B':
            i=0
def step_y():
    i = 1
    u = 0
    while i:
        while u!=1:
            u = ser.write( b'V') 
        u=0 
        ff=ser.read(1)
        #print(1111)
        if ff==b'B':
            i=0
step_right = int(0)
step_left = int(0)
def rotate_x(left, right):
    center = (right - left)/2 + left
    u = 0
    if center < 320:
        i=1
        while i:
            while u!=1:

                u = ser.write( b'r') 
            u=0 
            ff=ser.read(1)
            #print(1111)
            if ff==b'B':
                i=0
       # print("right")
    if center > 320:
        i=1
        while i:
            while u!=1:

                u = ser.write( b'l') 
            u=0 
            ff=ser.read(1)
            print(1111)
            if ff==b'B':
               i=0
        #print("left")
    global step_right
    global step_left
    if center > 360:
        
        step_right = step_right + 1
        step_left = step_left - 1
        if step_left < 0:
            step_left = 0
        if step_right < 30:
            step()
        else: step_right = 30
    if (center < 250 and center != 0):
        step_right = step_right - 1
        step_left = step_left + 1
        if step_right < 0:
            step_right = 0
        if step_left < 30:
            step()
        else: step_left = 30

def rotate_y(vniz, vverx):
    #if up == 0:
        #step_none()
    #else: step_y1(up)
    center = (vverx - vniz)/2 + vniz
    #center = 480 - vverx
    step_y1(center)

step_vverx = int()
step_vniz = int()
repeat = int()


def step_none():
    u = 0
    global step_vverx
    global step_vniz
    global repeat
    if (step_vverx != 41 and repeat == 0):

        i=1
        while i:
            while u!=1:

                u = ser.write( b'R') 
            u=0 
            ff=ser.read(1)
            #print(1111)
            if ff==b'B':
                i=0
        #print ("vverx")

        step_vverx = step_vverx + 1
        step_vniz = step_vniz - 1
        if step_vniz < 0:
            step_vniz = 0
        if step_vverx < 40:
            step_y()
        else: 
            step_vverx = 40
            repeat = 1

    if (step_vniz != 41 and repeat == 1):

        i=1
        while i:
            while u!=1:

                u = ser.write( b'L') 
            u=0 
            ff=ser.read(1)
            #print(1111)
            if ff==b'B':
                i=0
        #print ("vniz")

        step_vverx = step_vverx - 1
        step_vniz = step_vniz + 1
        if step_vverx < 0:
            step_vverx = 0
        if step_vniz < 40:
            step_y()
        else: 
            step_vniz = 40
            repeat = 0


def step_y1(center):
    global step_vverx
    global step_vniz
    u = 0
    if center < 300:
        i=1
        while i:
            while u!=1:

                u = ser.write( b'L') 
            u=0 
            ff=ser.read(1)
            #print(1111)
            if ff==b'B':
                i=0
    if center > 300:
        i=1
        while i:
            while u!=1:

                u = ser.write( b'R') 
            u=0 
            ff=ser.read(1)
            #print(1111)
            if ff==b'B':
                i=0
    if (center < 280 and center != 480 and center != 0):

        step_vverx = step_vverx + 1
        step_vniz = step_vniz - 1
        if step_vniz < 0:
            step_vniz = 0
        if step_vverx < 100:
            step_y()
        else: step_vverx = 100
    if center > 340:

        step_vverx = step_vverx - 1
        step_vniz = step_vniz + 1
        if step_vverx < 0:
            step_vverx = 0
        if step_vniz < 100:
            step_y()
        else: step_vniz = 100

    print(center)
h = 0
y1 = 0
while(True):
# Capture frame-by-frame
    ret, frame = cap.read()
    while ret != True:
        ret, frame = cap.read()
# Our operations on the frame come here
    n = 0
    detected_copy, detections = detector.detectObjectsFromImage(input_image=frame, input_type ="array", output_type = "array")
    for eachObject in detections:
        if eachObject["name"] == "person":
            str1 = eachObject["box_points"]
            str2= str(str1)
            n = 1
            for i in range(5):
                if str2[i] in [","]:
                    detect1 = int(str2[1:i])
                    #print(detect1)
            print (eachObject["box_points"])
    #cv2.imwrite("1.jpg", frame)

#list of x1,y1,x2 and y2
    #img = cv2.imread("image2new.jpg") 

    left = str1[0]
    right = str1[2]
    rotate_x(left, right)
    if n == 1:
        niz = str1[1]
        verx = str1[3]
    else: 
        verx = 0
        niz = 0
    rotate_y(niz, verx)
    #step_none()


    #time.sleep(1)
    #print(step_vverx, step_vniz)
    cv2.imshow("gggg", frame)
    if cv2.waitKey(33) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
