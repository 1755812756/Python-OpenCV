# 导入cv模块
import os
import cv2
import urllib
import urllib.request

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
# 加载数据
recogizer.read('K:/python/OpenCV/trainer/trainer.yml')

# 名称
names = []
# 警报全局变量
warningtime = 0


# md5加密
def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode('utf8'))
    return m.hexdigest()


# 短信反馈
statusStr = {}


# 报警模块
def warning():
    print('报警反馈')


# 准备识别的图片
def face_detect_demo(img):
    gary = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_detect = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        ids, confindence = recogizer.predict(gary[y:y + h, x:x + h])
        print(confindence)
        if confindence < 65:
            global warningtime
            warningtime += 1
            # 陌生人识别次数超过100次 进行报警
            if warningtime > 100:
                warning()
                warningtime = 0
            #cv2.putText(img, 'undows', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img, names[ids - 1], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)


def getname():
    path = 'C:/Users/Administrator/Desktop/face/'
    # names = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.', 2)[1])
        names.append(name)


# 读取摄像头
cap = cv2.VideoCapture('rtsp://192.168.2.111:554/av0_0')
getname()
# 读取视频
# cap = cv.VideoCapture('1.mp4')

c = 0
# 等待
while True:
    flag, frame = cap.read()
    if flag:
        if (c % 5 == 0):
            face_detect_demo(frame)
        c += 1
        cv2.waitKey(1)
    else:
        break
# 释放内存
cv2.destroyWindow()

# 释放摄像头
cap.release()
