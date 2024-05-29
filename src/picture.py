#!/usr/bin/env python
"""Class initialization
"""

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

class Picture:
    

    def __init__(self, path):
        """Initialize Picture class with path to .jpg

        Args:
            path (None): Path to picture
        """
        try:
            self.image_raw = cv2.imread(path)
            self.image_resized = cv2.resize(self.image_raw, (800,600), fx=0.0, fy=0.0, interpolation=cv2.INTER_LANCZOS4)
            self.image_gray = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2GRAY)
            self.image_hsv = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2HSV)

        except Exception as e:
            print(f"An error occurred: {e}")

    def empty_callback(*args):
        pass

    def preproccessing(self, develop:bool=False) -> None:        

        if develop:
            cv2.namedWindow('Parameters preproccessing')
            cv2.createTrackbar('Filter', 'Parameters preproccessing', 0, 10, self.empty_callback)
            cv2.createTrackbar('BIFilter', 'Parameters preproccessing', 0, 25, self.empty_callback)
            cv2.createTrackbar('sigmaCol', 'Parameters preproccessing', 0, 25, self.empty_callback)
            cv2.createTrackbar('sigmaSpac', 'Parameters preproccessing', 0, 25, self.empty_callback)
            cv2.createTrackbar('Erosion', 'Parameters preproccessing', 0, 10, self.empty_callback)
            cv2.createTrackbar('Dilation', 'Parameters preproccessing', 0, 10, self.empty_callback)
            cv2.createTrackbar('Opening', 'Parameters preproccessing', 0, 10, self.empty_callback)
            cv2.createTrackbar('Closing', 'Parameters preproccessing', 0, 10, self.empty_callback)
        
            while True: 
                image_work = self.image_gray 
                key_code = cv2.waitKey(10)
                if key_code == 27:
                    break
                track_fil = cv2.getTrackbarPos('Filter', 'Parameters preproccessing')
                track_bi = cv2.getTrackbarPos('BIFilter', 'Parameters preproccessing')
                track_bicol = cv2.getTrackbarPos('sigmaCol', 'Parameters preproccessing')
                track_bispa = cv2.getTrackbarPos('sigmaSpac', 'Parameters preproccessing')
                track_ero = cv2.getTrackbarPos('Erosion', 'Parameters preproccessing')
                track_dil = cv2.getTrackbarPos('Dilation', 'Parameters preproccessing')
                track_ope = cv2.getTrackbarPos('Opening', 'Parameters preproccessing')
                track_clo = cv2.getTrackbarPos('Closing', 'Parameters preproccessing')

                track_fil = int(2*track_fil-1)
                track_ero = int(2*track_ero-1)
                track_dil = int(2*track_dil-1)
                track_ope = int(2*track_ope-1)
                track_clo = int(2*track_clo-1)

                

                if track_fil > 0:
                    image_work = cv2.GaussianBlur(image_work, (track_fil, track_fil), 0)
                if track_bi > 0:
                    image_work = cv2.bilateralFilter(image_work, track_bi, track_bicol, track_bispa)

                image_work = cv2.Canny(image_work, 50, 200)

                if track_ero > 0:
                    kernel_ero = np.ones((track_ero, track_ero), np.uint8)
                    image_work = cv2.erode(image_work, kernel_ero, iterations=1)
                if track_dil > 0:
                    kernel_dil = np.ones((track_dil, track_dil), np.uint8)
                    image_work = cv2.dilate(image_work, kernel_dil, iterations=1)          
                if track_ope > 0:
                    kernel_ope = np.ones((track_ope, track_ope), np.uint8)
                    image_work = cv2.morphologyEx(image_work, cv2.MORPH_OPEN, kernel_ope)
                if track_clo > 0:
                    kernel_clo = np.ones((track_clo, track_clo), np.uint8)
                    image_work = cv2.morphologyEx(image_work, cv2.MORPH_CLOSE, kernel_clo)

                

                cv2.imshow('Parameters preproccessing', image_work)

        else:
            image = cv2.GaussianBlur(self.image_gray, (7, 7), 0)
            image = cv2.Canny(image, 50, 200)
            kernel_dil = np.ones((3, 3), np.uint8)
            self.image_preproccesed= cv2.dilate(image, kernel_dil, iterations=1)
            # kernel_ope = np.ones((3, 3), np.uint8)
            # self.image_preproccesed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_ope)
            
    
    def contouring(self) -> None:
        contours, _ = cv2.findContours(self.image_preproccesed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = self.image_resized.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.show()
        candidate_contours = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 5:
                    candidate_contours.append(approx)
        contour_image = self.image_resized.copy()
        self.filtered_contours = sorted(candidate_contours, key=cv2.contourArea, reverse=True)[:1]


    

    def masking(self) -> None:
        self.mask = np.zeros(self.image_gray.shape, np.uint8)
        self.new_image = cv2.drawContours(self.mask, self.filtered_contours, 0, 255, -1)
        self.new_image = cv2.bitwise_and(self.image_resized, self.image_resized, mask=self.mask)
        plt.imshow(cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB))
        plt.show()
        
    def croping_plate(self) -> None:
        try:
            (x,y) = np.where(self.mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = self.image_resized[x1:x2+1, y1:y2+1]
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.show()
        except:
            print("error")