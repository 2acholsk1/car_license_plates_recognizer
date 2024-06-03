#!/usr/bin/env python
"""Picture class script
"""

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import imutils


class Picture:
    """Class for initial preproccess images and extract license plates
    """
    

    def __init__(self, path, size_init:int):
        """Initialize Picture class with path to .jpg

        Args:
            path (None): Path to picture
        """
        try:
            self.raw_img = cv2.imread(path)
            img_ratio = self.raw_img.shape[1]/self.raw_img.shape[0]

            self.resized_img = cv2.resize(self.raw_img, (size_init, int(size_init/img_ratio)), fx=0.0, fy=0.0, interpolation=cv2.INTER_LANCZOS4)
            self.plate = None

        except Exception as e:
            print(f"An error occurred: {e} uno")



    def empty_callback(*args):
        pass

    def preproccessing(self, develop:bool=False) -> None:
        """Images preproccesing

        Args:
            develop (bool, optional): On off developer mode. Defaults to False.
        """

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
                blurred_img = cv2.GaussianBlur(self.resized_img, (7, 7), 0)
                image_work = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
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


    def blue_rectangle_finder(self, blue_min:np.ndarray, blue_max:np.ndarray):
        """Function to find left upper corner of blue rectangle from license plate

        Args:
            blue_min (np.ndarray): Minimal values of blue color in HSV
            blue_max (np.ndarray): Maximal values of blue color in HSV

        Returns:
            _type_: Coordinates x and y of blue rectangle corner
        """

        bilateral = cv2.bilateralFilter(self.resized_img.copy(), 3, 75, 75)
        hsv_img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv_img, blue_min, blue_max)

        contours = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_grab = imutils.grab_contours(contours)
        contours_sort = sorted(contours_grab, key=cv2.contourArea, reverse=True)[:10]
        
        for cnt in contours_sort:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w:
                return x, y
        
        return 300, 400

    def find_solidity(self, count):
        """Function to check solidity of contour

        Args:
            count (_type_): Contour to checj

        Returns:
            _type_: Value of solidity
        """
        contourArea = cv2.contourArea(count) 
        convexHull = cv2.convexHull(count) 
        contour_hull_area = cv2.contourArea(convexHull) 
        solidity = float(contourArea)/contour_hull_area 
        return solidity
    
    def contouring_plate(self, blue_min:np.ndarray, blue_max:np.ndarray, 
                         white_min:np.ndarray, white_max:np.ndarray,
                         min_area:int, max_area:int, width_plate:int, height_plate:int ) -> None:
        """Function for finding white plate rectangle. Based on mask with color in hsv

        Args:
            blue_min (np.ndarray): Minimal values of blue color in HSV
            blue_max (np.ndarray): Maximal values of blue color in HSV
            white_min (np.ndarray): Minimal values of white color in HSV
            white_max (np.ndarray): Maximal values of white color in HSV
            min_area (int): Minimal area of license plate rectangle
            max_area (int): Maximal area of license plate rectangle
            width_plate (int): Width of plate
            height_plate (int): Height of plate
        """

        x_cor, y_cor = self.blue_rectangle_finder(blue_min, blue_max)

        bilateral = cv2.bilateralFilter(self.resized_img.copy(), 3, 75, 75)
        hsv_img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv_img, white_min, white_max)
        

        contours = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_grab = imutils.grab_contours(contours)
        contours_sort = sorted(contours_grab, key=cv2.contourArea, reverse=True)[:10]

        for cnt in contours_sort:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            corners = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            solidity = self.find_solidity(cnt)

            if x > x_cor and min_area < area < max_area and len(corners) == 4 and 0.99 > solidity > 0.91:
                array_float = np.array([corners[0][0], corners[1][0],
                                        corners[2][0], corners[3][0]], dtype=np.float32)
                points = self.corners_matching(array_float)
                fit_img = np.float32([[0, 0], [width_plate, 0], [width_plate, height_plate], [0, height_plate]])
                matrix = cv2.getPerspectiveTransform(points, fit_img)
                self.plate = cv2.warpPerspective(self.resized_img.copy(), matrix, (width_plate, height_plate))

    def contouring_plate_mod2(self, blue_min, blue_max,
                              min_area:int, max_area:int,
                              width_plate:int, height_plate:int):
        """Function for finding white plate rectangle. Based on thresholding image with canny transformation


        Args:
            blue_min (np.ndarray): Minimal values of blue color in HSV
            blue_max (np.ndarray): Maximal values of blue color in HSV
            min_area (int): Minimal area of license plate rectangle
            max_area (int): Maximal area of license plate rectangle
            width_plate (int): Width of plate
            height_plate (int): Height of plate
        """

        x_cor, y_cor = self.blue_rectangle_finder(blue_min, blue_max)

        blurred_img = cv2.GaussianBlur(self.resized_img.copy(), (7, 7), 0)
        gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(gray_img, 50, 200)
        _, threshold_img = cv2.threshold(canny_img, 100, 255, cv2.THRESH_BINARY)
        closed_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, (5,5))
        kernel_dil = np.ones((3, 3), np.uint8)
        preproccessed_img = cv2.dilate(closed_img, kernel_dil, iterations=1)
        contours = cv2.findContours(preproccessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_grab = imutils.grab_contours(contours)
        contours_sort = sorted(contours_grab, key=cv2.contourArea, reverse=True)[:10]

        candidate_contours = []

        for cnt in contours_sort:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            corners = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            solidity = self.find_solidity(cnt)
            aspect_ratio = w / float(h)

            if x > x_cor and 1.5 <= aspect_ratio <= 5 and len(corners) == 4 and min_area < area < max_area and 0.99 > solidity > 0.91:
                candidate_contours.append(corners)
                
        filtered_contour = candidate_contours[:1]

        mask = np.zeros(gray_img.shape, np.uint8)
        new_image = cv2.drawContours(mask, filtered_contour, 0, 255, -1)
        new_image = cv2.bitwise_and(self.resized_img, self.resized_img, mask=mask)

        try:
            array_float = np.array([filtered_contour[0][0][0], filtered_contour[0][1][0],
                                    filtered_contour[0][2][0], filtered_contour[0][3][0]], dtype=np.float32)
            
            array_float = self.corners_matching(array_float.copy())
            fit_img = np.float32([[0, 0], [width_plate, 0], [width_plate, height_plate], [0, height_plate]])
            matrix = cv2.getPerspectiveTransform(array_float, fit_img)

            self.plate = cv2.warpPerspective(new_image, matrix, (width_plate, height_plate))
            
        except Exception as e:
            print(f"An error occurred: {e}")

    def corners_matching(self, array:np.array) -> None:
        """Function for correcting rotation of recognized license plate

        Args:
            array (np.array): Corners

        Returns:
            _type_: Proper corners order
        """
        array_sum_col = np.sum(array, axis=1)
        min_row = np.argmin(array_sum_col)
        max_row = np.argmax(array_sum_col)
        first_cor = array[min_row]
        third_cor = array[max_row]
        array_of_two = np.delete(array, [min_row, max_row], axis=0)
        greater_x = np.argmax(array_of_two[:,0])
        non_greater_x = np.argmin(array_of_two[:,0])
        second_cor = array_of_two[greater_x]
        fourth_cor = array_of_two[non_greater_x]
        return np.array([first_cor, second_cor, third_cor, fourth_cor], dtype=np.float32)
    
    def plate_get(self):
        """Get plate license image

        Returns:
            _type_: Image of license plate 
        """
        return self.plate
