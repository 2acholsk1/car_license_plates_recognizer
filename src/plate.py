#!/usr/bin/env python


import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


class Plate:

    def __init__(self, plate:np.ndarray) -> None:
        
        if plate is not None:
            self.plate_raw = plate
            self.plate_raw = cv2.resize(plate, (876, 236), cv2.INTER_AREA)
            self.min_letter_rec = 100
            border_width = 10
            self.plate_raw = cv2.copyMakeBorder(self.plate_raw, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(255,255,255))
        
    def empty_callback(*args):
        pass

    def preproccess(self, develop:bool=False):

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
                blurred_img = cv2.GaussianBlur(self.plate_raw, (5, 5), 0)
                gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
                image_work = gray_img 
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
                _, image_work = cv2.threshold(image_work, 100, 255, cv2.THRESH_BINARY)

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

        blurred_img = cv2.GaussianBlur(self.plate_raw, (5,5), 0)
        gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(gray_img, 50, 255)
        _, threshold_img = cv2.threshold(canny_img, 0, 255, cv2.THRESH_BINARY)
        kernel_dil = np.ones((9, 9), np.uint8)
        dilate_img = cv2.dilate(threshold_img, kernel_dil, iterations=1)
        plt.imshow(cv2.cvtColor(dilate_img, cv2.COLOR_BGR2RGB))
        plt.show()

        min_area = 12000
        max_area = 22500
        

        contours, _ = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
            
        #     if area > min_area and area < max_area:
        #         x,y,w,h = cv2.boundingRect(cnt)
        #         cv2.rectangle(self.plate_raw,(x,y),(x+w,y+h),(0,255,0),2)

        # for cnt in contours:
        #     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        #     area = cv2.contourArea(cnt)
        #     if len(approx) == 4:
        #         x, y, w, h = cv2.boundingRect(approx)
        #         aspect_ratio = w / float(h)
        #         if 1 <= aspect_ratio <= 5 and area > min_area and area < max_area:
        #             cv2.rectangle(self.plate_raw,(x,y),(x+w,y+h),(0,255,0),2)
        
        # filtered_contour = sorted(contours, key=cv2.contourArea, reverse=True)[1:9]
        cv2.drawContours(self.plate_raw, contours, -1, (0, 255, 0), 1)

        plt.imshow(cv2.cvtColor(self.plate_raw, cv2.COLOR_BGR2RGB))
        plt.show()