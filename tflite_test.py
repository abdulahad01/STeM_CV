import time
import numpy as np
import imutils
import cv2
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

err_det=True
# required variables and uploads
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = '/home/pi/Downloads/STeM_CV/model.tflite'
color = (255,100,100)


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
        #cv2.putText(frame, label, (15,20),cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)
        #cv2.imshow('Frame',frame)
        #key = cv2.waitKey(1)
      
#       press q to exit()
        #if key == ord('q'):
            #stream.stop()
            #break
    #cv2.destroyAllWindows()



if __name__ == '__main__':
  main()