import cv2
import cv2 as cv
import numpy as np
import math

def nothing(x):
    pass

drawing = False
(ix, iy) = (0, 0)
(fx, fy) = (150, 150)

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

cam = cv2.VideoCapture('testvideo.mp4')
ret_val, img_copy  = cam.read()
img_copy = cv2.resize(img_copy, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
cv.imshow('img_copy', img_copy)

cv.namedWindow('img_color')

cv.createTrackbar('S', 'img_color', 0, 255, nothing)
cv.createTrackbar('V', 'img_color', 0, 255, nothing)

def draw(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = x, y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        (fx, fy) = x, y
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 3)


def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    S = cv.getTrackbarPos('S', 'img_color')
    V = cv.getTrackbarPos('V', 'img_color')

    if event == cv.EVENT_LBUTTONDOWN:
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        if hsv[0] < 10:
            lower_blue1 = np.array([hsv[0]-10+180, S, V])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, S, V])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], S, V])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])

        elif hsv[0] > 170:
            lower_blue1 = np.array([hsv[0], S, V])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, S, V])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, S, V])
            upper_blue3 = np.array([hsv[0], 255, 255])

        else:
            lower_blue1 = np.array([hsv[0], S, V])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, S, V])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, S, V])
            upper_blue3 = np.array([hsv[0], 255, 255])


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)


while(True):
    cv2.setMouseCallback('img_copy', draw)
    if fx != 150 | fy != 150:
        ret_val, img = cam.read()
        img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

        ROI_img = img[iy:fy, ix:fx]

        img_color = cv.imread('spectrum.jpg')
        img_color2 = ROI_img
        height, width = img_color.shape[:2]
        img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

        img_hsv = cv.cvtColor(img_color2, cv.COLOR_BGR2HSV)

        img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
        img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
        img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
        img_mask = img_mask1 | img_mask2 | img_mask3

        img_result = cv.bitwise_and(img_color2, img_color2, mask=img_mask)


        cv.imshow('img_color', img_color)
        cv.imshow('img', img)
        cv.imshow('img_result', img_result)

        dst = cv.Canny(img_result, 50, 200, None, 3)

        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 140, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()