import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

camera = PiCamera()
raw = PiRGBArray(camera)
time.sleep(0.1)


camera.capture(raw, format='bgr')
image = raw.array

cv2.imshow('img',image)
cv2.waitKey(0)
#cv2.destroyAllWindows