# 导入cv模块
import cv2
import cv2 as cv

cap = cv.VideoCapture(0)
falg = 1
num = 1

while (cap.isOpened()):
    ret_flag,Vshow = cap.read()
    cv2.imshow("Capture_Test", Vshow)
    k = cv2.waitKey(1) & 0xFF  # 判断按键
    if k == ord('s'):  # 保存
        cv2.imwrite("C:/Users/Administrator/Desktop/face/" + str(num) + ".jpg", Vshow)
        print("OK")
        num += 1
    elif k == ord(' '):  # 退出
        break

# 释放内存
cv.destroyWindow()

# 释放摄像头
cap.release()
