import cv2
import os
from picture import Picture

def load_and_display_images(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        
        pic = Picture(image_path)
        pic.preproccessing()
        pic.contouring()
        pic.masking()
        pic.croping_plate()
    

folder_path = 'data/train_1'
load_and_display_images(folder_path)


