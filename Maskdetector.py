# import the necessary packages
import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from imutils.video import VideoStream
import imutils
import time
import cv2
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import datetime
from threading import Thread



class MaskDetector :
    def __init__(self,model,prototxtPath, weightsPath,cap = None):
#         self.masknet = tflite.Interpreter(model)        
        self.maskNet = load_model("detector1.model")
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.stream =  cap

        
    def detect_and_predict_mask(self,frame):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.75:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = self.maskNet.predict(faces, batch_size=32)
#                 preds = interpreter.get_signature_runner()


            # return a 2-tuple of the face locations and their corresponding
            # locations
        return (locs, preds)
        
    def capture (self):
        #initialize the video stream
        print("[INFO] starting video stream...")
#         vs = VideoStream(src=0).start()
        self.stream.start()


        # loop over the frames from the video stream
        while True:
            
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
#             self.frame = vs.read()
            frame = self.stream.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = self.detect_and_predict_mask(frame)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (withoutMask, mask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                self.stream.stop()
                break

            # do a bit of cleanup
        cv2.destroyAllWindows()
#         vs.stream.release()

class FPS:
    
    def __init__(self):
        #Starting time stamp
        # ending timestamp
        # Number of frames captured
        self.start = None
        self.end = None
        self._numFrames = 0
        
    def start(self):
        # Starts the timer
        self._start =datetime.datetime.now()
        return self
    
    def stop(self):
        self._end =datetime.datetime.now()
        
    def update(self):
        #increments the number of frames
        self._numFrames += 1
    
    def elapsed(self):
        return(self._end - self._start).total_seconds()
    
    def fps(self):
        return self._numFrames
    
class webcamCapture:
    
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
            
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed,self.frame) = self.stream.read()
            
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped= True
        
    
        
    

if __name__ == '__main__':
    

    
    print('Starting script')
    
#     stream = webcamCapture()
#     stream.start()
#     while True:
#         frame = stream.read()
#         frame = imutils.resize(frame, width=400)
#         cv2.imshow('frame',frame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             stream.stop()
#             break
#     cv2.destroyAllWindows()
    
#     stream = webcamCapture()
#     prototxtPath = r"deploy.prototxt"
#     weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
#     model = r"detector1.model"
#     
#     detector = MaskDetector(model,prototxtPath, weightsPath, cap =stream)
#     detector.capture()
        
        
    
    
    
