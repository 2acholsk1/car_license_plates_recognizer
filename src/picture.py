#!/usr/bin/env python
"""Class initialization
"""

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

class Picture:
    

    def __init__(self, path, min_area, max_area):
        """Initialize Picture class with path to .jpg

        Args:
            path (None): Path to picture
        """
        try:
            self.raw_img = cv2.imread(path)
            self.resized_img = cv2.resize(self.raw_img, (1000,800), fx=0.0, fy=0.0, interpolation=cv2.INTER_LANCZOS4)
            self.min_area = min_area
            self.max_area = max_area

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
            blurred_img = cv2.GaussianBlur(self.resized_img, (7, 7), 0)
            self.gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
            canny_img = cv2.Canny(self.gray_img, 50, 200)
            _, threshold_img = cv2.threshold(canny_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            closed_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, (5,5))
            kernel_dil = np.ones((3, 3), np.uint8)
            self.preproccesed_img = cv2.dilate(closed_img, kernel_dil, iterations=1)
            plt.imshow(cv2.cvtColor(self.preproccesed_img, cv2.COLOR_BGR2RGB))
            plt.show()
            # kernel_ope = np.ones((3, 3), np.uint8)
            # self.image_preproccesed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_ope)
            
    
    def contouring(self) -> None:
        contours, _ = cv2.findContours(self.preproccesed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidate_contours = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 5 and self.check_contour(w, h):
                    candidate_contours.append(approx)
        self.filtered_contour = sorted(candidate_contours, key=cv2.contourArea, reverse=True)[:1]

    def check_contour(self, width, height) -> None:

        min = self.min_area 
        max = self.max_area 
   
        area = width*height
   
        if (area < min or area > max): 
            return False
           
        return True

    

    def masking(self) -> None:
        self.mask = np.zeros(self.gray_img.shape, np.uint8)
        self.new_image = cv2.drawContours(self.mask, self.filtered_contour, 0, 255, -1)
        self.new_image = cv2.bitwise_and(self.resized_img, self.resized_img, mask=self.mask)
        
    def croping_plate(self) -> None:
        try:
            (x,y) = np.where(self.mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            self.cropped_img = self.resized_img[x1:x2+1, y1:y2+1]
            plt.imshow(cv2.cvtColor(self.cropped_img, cv2.COLOR_BGR2RGB))
            plt.show()
        except:
            print("error")


    def change_perspective(self) -> None:

        width, height, _= self.cropped_img.shape

        plate_blur = cv2.GaussianBlur(self.cropped_img, (7, 7), 0)
        plate_gray = cv2.cvtColor(plate_blur, cv2.COLOR_BGR2GRAY)
        image_thresh = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
        image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        contours, _ = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        
        for cnt in contours:
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))


        if len(centers) >= 4:
            centers = np.float32(centers[:4])
            print(len(centers))
            dst_points = np.float32([(width, height), (0, height), (width, 0), (0, 0)])
        
            trans_matrix = cv2.getPerspectiveTransform(np.float32(centers), dst_points)
            image_persp = cv2.warpPerspective(self.cropped_img, trans_matrix, (width, height))
            plt.imshow(cv2.cvtColor(image_persp, cv2.COLOR_BGR2RGB))
            plt.show()
