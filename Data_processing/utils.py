import os
import requests
import glob
import cv2
import random
import numpy as np
IMAGE_PATH = '/Users/hit.flouxetine/Desktop/NBA_2k/dataset/Images/'
LABEL_PATH = '/Users/hit.flouxetine/Desktop/NBA_2k/dataset/Labels/'
WIDTH = 1280
HEIGHT = 720
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


def shuffle(a):
    return random.shuffle(a)


def get_image(data_dict, file_name):
    image_url = data_dict["Labeled Data"]
    res = requests.get(image_url, timeout=20)
    with open(file_name, "wb") as f:
        f.write(res.content)


def get_label(data_dict, path, label):
    index = 1
    path = path + label + "/"
    if os.path.exists(path) is False:
        os.makedirs(path)
    try:
        for obj in data_dict["Label"]["objects"]:
            if obj["value"] == label:
                img = requests.get(obj["instanceURI"], timeout=20)
                with open(path + str(index) + ".png", "wb") as f:
                    f.write(img.content)
                index += 1
    except:
        print("there is no object")


def load_label(image_name, label_dict):
    '''
     params: a dictionary {"foul_space":0, "1point":1, "2point":2, "3point":3,...} describes the labels with index.
     params: image name
    '''
    label_path = LABEL_PATH + image_name + "/"
    label = []
    for label_class in sorted(os.listdir(label_path)):
        tmp_path = os.path.join(label_path, label_class)
        image_class_i = []
        for elem in sorted(os.listdir(tmp_path)):
            image = cv2.imread(os.path.join(tmp_path, elem))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_class_i.append(image)
        if len(image_class_i) != 0:
            image_class_i = np.array(image_class_i)
            image_class_i = image_class_i.sum(axis=0, dtype=np.uint8)
            mask = np.where(image_class_i > 0)
            image_class_i[mask] = label_dict[label_class]
        else:
            image_class_i = np.zeros(shape=(HEIGHT, WIDTH))
        label.append(image_class_i)
    label = np.array(label)
    label = label.max(axis=0)
    return label

def load_mask(image_name):
    mask = cv2.imread("./dataset/Masks/" + image_name + ".png")
    mask = mask[:, :, 0]
    return mask

def process_mask(output, COLOR):
    mask = np.zeros(shape=(output.shape[0], output.shape[1], 3))

    mask[output == 0] = COLOR[0]
    mask[output == 1] = COLOR[1]
    mask[output == 2] = COLOR[2]
    mask[output == 3] = COLOR[3]
    mask[output == 4] = COLOR[4]
    mask[output == 5] = COLOR[5]
    mask[output == 6] = COLOR[6]
    mask[output == 7] = COLOR[7]
    mask = np.reshape(mask, (720, 1280, 3))

    return mask