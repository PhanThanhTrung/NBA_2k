import cv2
import glob
import numpy as np
from utils import load_label
JSON_PATH = "./dataset.json"
IMAGE_PATH = "./dataset/Images/"
LABEL_PATH = "./dataset/Labels/"
SAMPLE_LABEL_DICT = {
    "foul_space": 1,
    "1point": 2,
    "2point": 3,
    "3point": 4,
    "player": 5,
    "ball": 6,
    "basket": 7,
    "background": 0
}

image_list = glob.glob(IMAGE_PATH + "*")
X = []
y = []
for image_path in image_list:
    image_name = image_path.split("/")[-1][:-4]
    print("[INFO] Loading on image path: ", image_name)
    label = load_label(image_name, SAMPLE_LABEL_DICT)
    cv2.imwrite("./dataset/Masks/"+image_name+".jpg",label)