#!/usr/bin/env python3
"""This script include main function of program
"""

import os
import sys
import json
from src.picture.picture import Picture
from src.plate.plate import Plate
import numpy as np
from src.config_func import load_config

pictures_path = sys.argv[1]
saving_path = sys.argv[2]

def main():
    """License Plate recognition main function
    """
    
    config_pic = load_config("config/picture_config.yaml")
    config_pla = load_config("config/plate_config.yaml")

    results = {}

    images = [f for f in os.listdir(pictures_path) if os.path.isfile(os.path.join(pictures_path, f))]
    
    for image_name in images:
        image_path = os.path.join(pictures_path, image_name)
        
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
        
        plate = Plate(pic.plate_get())
        plate.chars_recognize('data/font',
                                config_pla['CHAR']['width'], config_pla['CHAR']['height'],
                                config_pla['CHAR']['height_min'], config_pla['CHAR']['width_max'])

        results[image_name] = plate.text_get()
    
    with open(saving_path+'.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()