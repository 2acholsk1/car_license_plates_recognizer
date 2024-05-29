import cv2
import numpy as np

def preproccess():


    def is_within_angle_limit(approx, max_angle_deviation=45):
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            delta_y = abs(p2[1] - p1[1])
            delta_x = abs(p2[0] - p1[0])
            if delta_x == 0:
                return False
            angle = np.degrees(np.arctan(delta_y / float(delta_x)))
            if angle > max_angle_deviation:
                return False
        return True


    image = cv2.imread('data/train_1/PSZ47620.jpg')
    scaling = cv2.resize(image, (800,600), fx=0.0, fy=0.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(scaling, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    dilate = cv2.dilate(edged, (5,5), iterations=2)
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, (5,5))
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 <= aspect_ratio <= 5:
                candidate_contours.append(approx)
    
    cv2.drawContours(scaling, candidate_contours, -1, (0, 255, 0), 2)

    cv2.namedWindow('Test')
    cv2.namedWindow('Original')

    while True:
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break
        cv2.imshow('Original', scaling)
        cv2.imshow('Test', edged)

preproccess()

