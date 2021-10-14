import cv2
cap = cv2.VideoCapture(0)
while 1:
    ret,frame = cap.read()
    if ret :
        cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
    