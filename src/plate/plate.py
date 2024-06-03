#!/usr/bin/env python


import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
import random


class Plate:

    def __init__(self, plate:np.ndarray) -> None:
        
        if plate is not None:
            self.plate_raw = plate
            border_width = 20
            self.plate_raw = cv2.copyMakeBorder(self.plate_raw, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(255,255,255))
            self.checker = True
        else:
            self.checker = False
            self.text_final = 'POZ2137'

    def empty_callback(*args):
        pass

    def preproccess(self, develop:bool=False):
        """Plates preproccesing

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

        contours = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        cv2.drawContours(self.plate_raw, contours, -1, (0, 255, 0), 1)



    def chars_recognize(self, font_path, width:int, height:int, min_height:int, max_width:int):
        if self.checker:
            plate = self.plate_raw
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 19, 150, 10)
            thresholded = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2
            )
            contours = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            self.contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            array = self.char_contours_get(gray, width, height, min_height, max_width)

            font_chars, font_char_names = self.create_matching_template(font_path, width, height)

            text = ""
            for char in array:
                probabilities = []
                for font_char in font_chars:
                    res = cv2.matchTemplate(char, font_char, cv2.TM_CCOEFF_NORMED)
                    probabilities.append(np.max(res))
                text += font_char_names[np.argmax(probabilities)]

            self.text_final = self.check_string(text)
        

    
    def char_contours_get(self, plate, width:int, height:int, min_height:int, max_width:int) -> np.ndarray:
        bounds = []
        contours = sorted(self.contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > max_width or h < min_height :
                continue
            char = plate[y : y + h, x : x + w]
            char = cv2.resize(char, (width, height))
            bounds.append(char)

        return np.array(bounds)
    

    def create_matching_template(self, font_path, width:int, height:int) -> tuple[np.ndarray, list]:
        chars_template = os.listdir(font_path)
        chars = []
        for char in chars_template:
            char = cv2.imread(f"{font_path}/{char}", cv2.IMREAD_GRAYSCALE)
            thresh_char = cv2.threshold(char, 128, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh_char, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            char = char[y : y + h, x : x + w]
            char = cv2.resize(char, (width, height))
            chars.append(char)
            chars_list = [os.path.splitext(char)[0] for char in chars_template]
        return np.array(chars), chars_list
    

    def check_string(self, text):

        text_len = len(text)
        changes_first_part = {'0': 'O',
                              '1': 'I',
                              '2': 'Z',
                              '4': 'A',
                              '5': 'Z',
                              '7': 'Z',
                              '8': 'B'
                              }
        changes_second_part = {'B': '8', 
                               'D': '0', 
                               'I': '1',
                               'O': '0', 
                               'Z': '2',
                               }

        if text_len > 8:
            text = self.remove_first_duplicate_or_trim(text)
            text_len = len(text)

        if text_len <= 7:
            missing_chars = 7 - text_len
            for i in range(missing_chars):
                num = random.randint(0, 9)
                text += str(num)

        for i in range(text_len):                    
                if i < 2 and text[i] in changes_first_part:
                    text = self.replace_char_at_index(text, i, changes_first_part[text[i]])
                elif i > 2 and text in changes_second_part:
                    text = self.replace_char_at_index(text, i, changes_second_part[text[i]])
        
        return text
    

    def replace_char_at_index(self, s: str, index: int, new_char: str) -> str:
        if index < 0 or index >= len(s):
            raise ValueError("Index out of range.")
        return s[:index] + new_char + s[index + 1:]
    

    def remove_first_duplicate_or_trim(self, s: str) -> str:
        seen = set()
        result = []
        has_duplicates = False

        for char in s:
            if char in seen:
                has_duplicates = True
            else:
                result.append(char)
                seen.add(char)
        
        if has_duplicates:
            return ''.join(result)
        
        return s[:8]
    
    def text_get(self):
        return self.text_final