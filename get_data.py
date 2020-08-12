import json
from utils import *
import glob

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
if __name__ == "__main__":
    with open(JSON_PATH) as json_file:
        data = json.load(json_file)

    index = 1
    for data_dict in data:
        print("[INFO] Working on image", index)
        image_file_path = IMAGE_PATH + "image" + str(index) + ".png"
        label_path = LABEL_PATH + "image" + str(index) + "/"
        get_image(data_dict, image_file_path)
        get_label(data_dict, label_path, label="1point")
        get_label(data_dict, label_path, label="foul_space")
        get_label(data_dict, label_path, label="2point")
        get_label(data_dict, label_path, label="3point")
        get_label(data_dict, label_path, label="basket")
        get_label(data_dict, label_path, label="ball")
        get_label(data_dict, label_path, label="player")
        index += 1