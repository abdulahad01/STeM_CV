from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep
import numpy as np
import imutils
import cv2
import datetime
from threading import Thread
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

err_det=True
# required variables and uploads
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = 'model.tflite'
color = (255,100,100)
flag =0
temp=0

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

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]   
    global err_det
    try:
        input_tensor[:, :] = image
        err_det=True
        print("det")
    except:
        err_det=False
        print("no_det")

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    
    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

    
def detect_face(frame, faceNet):
    ''' isolate faces from the given frame'''
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
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
        if confidence > 0.60:
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
        faces = np.array(faces, dtype="float32")

    # return a 2-tuple of the faces and their corresponding
    # locations
    return (faces, locs)


def main():
    checker_count=0
    mask_checker=[]
    # load our serialized face detector model from disk
    interpreter = Interpreter(model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    stream = webcamCapture()
    print("[INFO] starting video stream...")
    stream.start()
    while True:
        frame = stream.read()
        frame = imutils.resize(frame,width=400)
        (image,locs) = detect_face(frame,faceNet)
        results = classify_image(interpreter,image)
        #       print(results)
        
        if(err_det==True):
            for val, pred in results:
                if (val == 0 and pred >=0.6):
                    label = 0  #no mask 
                elif val==1 and pred >=0.65:
                    label = 1 #mask
                checker_count=checker_count+1
                mask_checker.append(label)
                print(label)
                print(checker_count)
        
        elif(err_det==False):
            continue
        
        cv2.putText(frame, str(label),(15,20),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.imshow("MASK_det",frame)
        
        if checker_count==10:
            checker=0
            if sum(mask_checker)>=7:
                mask_checker=[]
                return True
            else:
                print("Please put your mask properly")
                checker_count=0
                continue
        else:
            continue

        
def temp_read():
    tmp=0
    temp_ary=[]
    for i in range(100):
        tmp= float(sensor.get_object_1())
        temp_ary.append(temp)
        print(max(temp_ary))
    return tmp
def temp_over_limit(temperature):
    while(temperature>=36.8):
        temperature= temp_read()
        print("Temperature over the limit ,\n'''Inform office'''")
    
#Pin 25 sanitizer valve
#Pin 17 for moving rod up
#Pin 27 for moving rod down
#Pin 22 for stopping the rod moving down
#Pin 23 for stopping the rod moving up
#Pin 4 Pin for the IR sensor

GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme
GPIO.setwarnings(False)
GPIO.setup(4, GPIO.IN) # ir pin set as input w/ pull-up
GPIO.setup(23, GPIO.IN) #kill_switch for stopping the opening flap
GPIO.setup(22, GPIO.IN)#kill_switch for stopping the closing flap
GPIO.setup(25, GPIO.OUT)#sanitizer valve activation (signal to MOSFET)
GPIO.setup(17, GPIO.OUT)#flap opening command initiating pin
GPIO.setup(27, GPIO.OUT)#flap closing command initiating pin

GPIO.output(17,0)
GPIO.output(27,0)

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
count=0
f=1
while (f==1):
    if(GPIO.input(4)==0):
        temp = temp_read()
        while(temp_read()<=33):
            flag=0
            rod_state=0
            print("please place your hand again")
        GPIO.output(25,1)
        sleep(0.2)#delay in seconds(sanitiser dispensing)
        GPIO.output(25,0)
        sleep(0.5)
        #********normal temperature***************
        if temp<=35 and temp>=32:
            flag=1
            
        #********over limit temperature***************
        elif temp>36.8:
            flag=0
            temp_over_limit(temp)
            break
        #********under limit temperature***************   
        
        #mask detection to be initiated here
        mask_det=main()
        if mask_det==True and flag==1:
            rod_state=1#if motor_r_state==1 flap in motion otherwise not in motion
        #opening the flap
        else:
            rod_state=0
        count=0
        print('start')
        while(GPIO.input(23)==0 or count<=50):
            GPIO.output(17,rod_state)#initiated command to rod (motor) to open the way
            sleep(0.1)
            count+=1
            print("o")
        GPIO.output(17,0)
        print('next')
        #closing the flap
        count=0# for testing
        while(GPIO.input(22)==0 or count<=50):
            GPIO.output(27,rod_state)#initiated command to rod (motor) to open the way
            sleep(0.1)
            print("c")
            count+=1
            
        GPIO.output(27,0)
        print("Individual has been satisfied with all the parameters")#simply for rasam :) heheheheee
        f=0