import cv2
import os
from picture import Picture
from plate import Plate
import numpy as np
from src.config_func import load_config

def load_and_display_images(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    config = load_config("config/picture_config.yaml")
    
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        
        pic = Picture(image_path, config['PICTURE']['size'])
        # pic.preproccessing()
        pic.blue_rectangle_finder(np.array([config['BLUE']['h_min'], config['BLUE']['s_min'], config['BLUE']['v_min']]),
                                  np.array([config['BLUE']['h_max'], config['BLUE']['s_max'], config['BLUE']['v_max']]))
        
        pic.contouring_plate(np.array([config['BLUE']['h_min'], config['BLUE']['s_min'], config['BLUE']['v_min']]),
                             np.array([config['BLUE']['h_max'], config['BLUE']['s_max'], config['BLUE']['v_max']]),
                             np.array([config['WHITE']['h_min'], config['WHITE']['s_min'], config['WHITE']['v_min']]),
                             np.array([config['WHITE']['h_max'], config['WHITE']['s_max'], config['WHITE']['v_max']]),
                             config['PICTURE']['area_min'], config['PICTURE']['area_max'],
                             config['PICTURE']['width_plate'], config['PICTURE']['height_plate'])
        # pic.contouring()
        # pic.masking()
        # pic.croping_plate()
       
        # plate = Plate(pic.plate_get())
        # plate.preproccess(develop=False)
    

folder_path = 'data/train_1'
load_and_display_images(folder_path)


