# 导入cv模块
import cv2 as cv


# 检测函数
def face_detect_demo(img):
    gary = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)


# 读取摄像头
cap=cv.VideoCapture(0)

# 读取视频
# cap = cv.VideoCapture('1.mp4')

c=0
# 等待
while True:
    flag, frame = cap.read()
    if flag:
        if (c % 5 == 0):
            face_detect_demo(frame)
        c += 1
        cv.waitKey(1)
    else:
        break
# 释放内存
cv.destroyWindow()

# 释放摄像头
cap.release()
