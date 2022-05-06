#导入cv模块
import cv2 as cv
#读取图片
img =cv.imread('face1.jpg')
#检测函数
def face_detect_demo():
    gary=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    face_detect=cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face=face_detect.detectMultiScale(gary,1.1,5,0,(10,10),(100,100))
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)

face_detect_demo()
#等待
while True:
    if ord('q')==cv.waitKey(0):
        break
cv.waitKey(0)
#释放内存
cv.destroyWindow()