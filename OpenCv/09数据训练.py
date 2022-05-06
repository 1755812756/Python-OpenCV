# 导入cv模块
import os
import cv2
from PIL import Image
import numpy as np


def getImageAndLabels(path):
    # 保存人脸数据
    facesSamples = []
    # 存储姓名数据
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    # 编列列表中的图片
    for imaagePath in imagePaths:
        # 打开图片，灰度化 PIL 有九种不同模式1，L，p，RGB，RGBA，CMYK，ycbCr，I，F
        PIL_img = Image.open(imaagePath).convert('L')
        # 将 图像转为数组 以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的Id和姓名
        id = int(os.path.split(imaagePath)[1].split('.')[0])
        # 预防无面容照片
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:x + h, x:x + w])
    print('id', id)
    print('fs', facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = 'C:/Users/Administrator/Desktop/face/'
    # 获取图像数组和id标签数组和姓名
    faces, ids = getImageAndLabels(path)
    # 加载识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))
    # b保存文件
    recognizer.write('K:/python/OpenCV/trainer/trainer.yml')
