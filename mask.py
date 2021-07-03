import cv2
import numpy as np
print(cv2.version)
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
cv2.imshow("output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('copy_C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg',img)
#reading and displaying video
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc('X','V','I','D')
out=cv2.VideoWriter('Name of output file.avi',fourcc,20.0,(640,480))
while True:
    success,frame=cap.read()
    if success==True:
        out.write(frame)
        cv2.imshow("video",frame)
        if cv2.waitKey(1)==ord('q'):#we have to press q in order to stop video
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
#reading/displaying a video
cap= cv2.VideoCapture(0)
cap.set(3,640)#width
cap.set(4,480)#height
cap.set(10,100)#brightness
while True:
    success,img=cap.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1)&0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#converting the image into gray scale
#reading an image
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
imgGray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow('GrayScale Image',imgGray)
#imgGray1=cv2.cvtColor(imgGray,cv2.COLOR_GRAYSCALE)

cv2.waitKey(0)
cv2.destroyAllWindows()
#Blur function
#reading an image
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
imgBlur=cv2.GaussianBlur(img,(21,21),0)
cv2.imshow('Blurred Image',imgBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()
#edge detector -canny
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
imgCanny=cv2.Canny(img,190,250)
kernel=np.ones((5,5),np.uint8)
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
imgCanny=cv2.Canny(img,190,250)
kernel=np.ones((5,5),np.uint8)
imgDilation=cv2.dilate(imgCanny,kernel,iterations=1)
imgErosion=cv2.erode(imgDilation,kernel,iterations=1)
cv2.imshow('Canny Image',imgCanny)
cv2.imshow('Dilation Image',imgDilation)
cv2.imshow('Erosion Image',imgErosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
cv2.imshow("output",img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
#resize
img_resize=cv2.resize(img,(200,400))
cv2.imshow("original image",img)
cv2.imshow("Re-sized image",img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img=cv2.imread('C:/Users/HP/Pictures/swati.jpg')
cv2.imshow('output',img)
imgCropped=img[0:200,300:500]
cv2.imshow("cropped image",imgCropped)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#0 mean black
img=np.zeros((512,512))
cv2.imshow('0',img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
#0 means black
img=np.zeros((512,512,3),np.uint8)#print black img with 3 channles rgb
cv2.line(img,(0,0),(300,300),(0,255,255),3)
cv2.imshow('0',img)
#print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.rectangle(img,(0,0),(250,250),(0,255,255),cv2.FILLED) #is used to fill the rectangle
cv2.imshow('0',img)
#print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.circle(img,(400,50),30,(0,255,255),cv2.FILLED)
cv2.imshow('0',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.putText(img,"OpenCV",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
cv2.imshow("0",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=cv2.imread('C:/Users/HP/Pictures/1614507782426_-1135794380_images (4).jpeg')
imgHor=np.hstack((img,img))
cv2.imshow("Horizontal",imgHor)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
faceCascade=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
img=cv2.imread('C:/Users/HP/Pictures/swati.jpg')
img=cv2.resize(img,(400,350))
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("Output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cap=cv2.VideoCapture(0)
faceCascade=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
while True:
    success,frame=cap.read()
    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x+y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("Video",frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()"""

import numpy as np
import cv2
cap=cv2.VideoCapture("C:/Users/HP/Downloads/videoplayback (34).mp4")
faceCascade=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_eye..xml")
faceCascade1=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
while(True):
    success,frame=cap.read()
    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade1.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("video",frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    
cap=cv2.VideoCapture("C:/Users/HP/Downloads/videoplayback (34).mp4")
faceCascade=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_eye.xml")
faceCascade1=cv2.CascadeClassifier("C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
while(True):
    success,frame=cap.read()
    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=faceCascade.detectMultiScale(imgGray,1.1,4)
    faces=faceCascade1.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
    cv2.imshow("video",frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 

import tensorflow as tf
#print('2')
#print(tf.version.VERSION)
from tensorflow.keras.models import load_model
detector=load_model(r'C:\Users\HP\Documents\dummy.model')
import tensorflow as tf
import numpy
import cv2
cap=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier(r'C:/Users/HP/Documents/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
while True:
    (success, frame) = cap.read()  #reading the frame from the stream 
    new_image = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1)) #resizing the frame to speed up the process of detection
    face = classifier.detectMultiScale(new_image) #detecting faces from the frame(ROI)
    for x,y,w,h in face:
        try:
            face_img = new_image[y:x+h, x:x+w] #getting the coordinates for the face detected
            resized= cv2.resize(face_img,(224,224)) #resizing the  face detected to fit into the model in the shape(224,224)
            image_array = tf.keras.preprocessing.image.img_to_array(resized) #converting the detected image into an array 
            image_array = tf.expand_dims(image_array,0) #expanding the dimensions to fit in the model
            predictions = detector.predict(image_array) #making predictions on the ROI
            score = tf.nn.softmax(predictions[0]) #getting the results 
            label = numpy.argmax(score)
        except Exception as e:
            print('bad frame')
            
        if label == 0:
            cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(new_image," mask",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0), 2)
        elif label == 1:
            cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(new_image,'no_mask',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2)
        else:
            None
    #displaying the window after predicting the outcome
    cv2.imshow('face_window', new_image)
    
    #print(numpy.argmax(score), 100*numpy.max(score))
    #waitkey to terminate the loop
    key = cv2.waitKey(10) 
    if key == ord('q'):
        break
#release the stream 
cap.release()
cv2.destroyAllWindows()
