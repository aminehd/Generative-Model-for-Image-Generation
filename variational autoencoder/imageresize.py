import argparse
import cv2
import numpy as np
import glob
import os
import scipy
import matplotlib.pyplot as plt

//change below values to resize
new_img_width = 32
new_img_height = 32

def load_image(path):
    ''' Resize image to new_img_width X new_img_width and shuffle axis to create 3 arrays (RGB) '''
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (new_img_width, new_img_height))) / 127.5 - 1
    img = np.rollaxis(img, 2, 0)
    return img


input_path = "/home/vishal/Dropbox/SFU/ML/Project/data/celebA/test"
output_path = "/home/vishal/Dropbox/SFU/ML/Project/data/celebC/test"
paths = glob.glob(os.path.join(input_path, "*.jpg"))

IMAGES = np.array( [ load_image(p) for p in paths ] )
generated_images = [np.rollaxis(img, 0, 3) for img in IMAGES]
for index, img in enumerate(generated_images):
	face_file_name = "{}.jpg".format(index+1)
	cv2.imwrite(os.path.join(output_path, face_file_name), np.uint8(255 * 0.5 * (img + 1.0)))

