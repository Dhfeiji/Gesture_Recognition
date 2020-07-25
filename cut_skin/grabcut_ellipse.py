import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import os.path

def grabcut(image):
    img = cv2.imread(i)
    # 划定前景区域
#     rect = (100, 80, 180, 450)
    rect = (1, 1, img.shape[1], img.shape[0])
    mask = np.zeros(img.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    # mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    # np.where(condition, x, y)满足条件(condition)，输出x，不满足输出y。
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    out = img * mask2[:, :, np.newaxis]
    # opencv的像素是BGR顺序，然而matplotlib所遵循的是RGB顺序
    out1 = out[:, :, ::-1]
    # array转换成image
    out1 = Image.fromarray(out1)
    out1.save(os.path.join(path_save_grabcut, os.path.basename(i)))


def ellipse_detect(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)
    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    skin1 = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
                cv2.bitwise_not(skin, skin1)
    # cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    # cv2.imshow(image, img)
    dst = cv2.bitwise_and(img, img, mask=skin1)
    # cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)

    # array转换成image
    dst = dst[:, :, ::-1]
    dst = Image.fromarray(dst)
    dst.save(os.path.join(path_save_skin, os.path.basename(image)))
    # cv2.imshow("cutout", dst)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # 获取指定目录下的所有图片
    list = glob.glob(r"E:/cloth/fd/*.jpg")
    list1 = glob.glob(r"E:/cloth/fd-cut/*.jpg")

    path_save_grabcut = 'E:/cloth/fd-cut'
    path_save_skin = 'E:/cloth/fd-skin'

    # for i in list:
    #     grabcut(i)
    for j in list1:
        ellipse_detect(j)
