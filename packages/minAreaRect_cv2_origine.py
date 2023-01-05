import cv2
import numpy as np

image = cv2.imread('./crosswalk/1.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''
for c in contours:
    # 找到边界坐标
'''
'''
x, y, w, h = cv2.boundingRect(contours)  # 计算点集最外面的矩形边界
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
'''
contours=np.array(contours)
# 找面积最小的矩形
rect = cv2.minAreaRect(contours)
# 得到最小矩形的坐标
box = cv2.boxPoints(rect)
# 标准化坐标到整数
box = np.int0(box)
# 画出边界
cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
    
'''
    # 计算最小封闭圆的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 换成整数integer
    center = (int(x),int(y))
    radius = int(radius)
    # 画圆
    cv2.circle(image, center, radius, (0, 255, 0), 2)
'''

cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
cv2.imshow("img", image)
cv2.imwrite("./img_1.jpg", image)
