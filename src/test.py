import cv2
import os
from picture import Picture
from plate import Plate
import numpy as np
from src.config_func import load_config

def load_and_display_images(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    config_pic = load_config("config/picture_config.yaml")
    config_pla = load_config("config/plate_config.yaml")
    
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        
        pic = Picture(image_path, config_pic['PICTURE']['size'])
        
        pic.contouring_plate(np.array([config_pic['BLUE']['h_min'], config_pic['BLUE']['s_min'], config_pic['BLUE']['v_min']]),
                             np.array([config_pic['BLUE']['h_max'], config_pic['BLUE']['s_max'], config_pic['BLUE']['v_max']]),
                             np.array([config_pic['WHITE']['h_min'], config_pic['WHITE']['s_min'], config_pic['WHITE']['v_min']]),
                             np.array([config_pic['WHITE']['h_max'], config_pic['WHITE']['s_max'], config_pic['WHITE']['v_max']]),
                             config_pic['PICTURE']['area_min'], config_pic['PICTURE']['area_max'],
                             config_pic['PICTURE']['width_plate'], config_pic['PICTURE']['height_plate'])
        if pic.plate is None:
            pic.contouring_plate_mod2(np.array([config_pic['BLUE']['h_min'], config_pic['BLUE']['s_min'], config_pic['BLUE']['v_min']]),
                             np.array([config_pic['BLUE']['h_max'], config_pic['BLUE']['s_max'], config_pic['BLUE']['v_max']]),
                             config_pic['PICTURE']['area_min'], config_pic['PICTURE']['area_max'],
                             config_pic['PICTURE']['width_plate'], config_pic['PICTURE']['height_plate'])
        try:
            plate = Plate(pic.plate_get())
            # plate.preproccess(develop=False)
            plate.chars_recognize('data/font',
                                  config_pla['CHAR']['width'], config_pla['CHAR']['height'],
                                  config_pla['CHAR']['height_min'], config_pla['CHAR']['width_max'])
        except Exception as e:
            print(f"An error occurred: {e}")
    

folder_path = 'data/train_1'
load_and_display_images(folder_path)


