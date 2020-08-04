import os
import glob
import cv2
import numpy as np

IMAGE_PATH = './dataset/Images/'
LABEL_PATH = './dataset/Labels/'
WIDTH = 1280
HEIGHT = 720
SAMPLE_LABEL_DICT = {"foul_space": 1, "1point": 2, "2point": 3,
                     "3point": 4, "player": 5, "ball": 6, "basket": 7, "background": 0}


def load_label(image_name, label_dict):
    '''
     params: a dictionary {"foul_space":0, "1point":1, "2point":2, "3point":3,...} describes the labels with index.
     params: image name
    '''
    label_path = LABEL_PATH+image_name+"/"
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
    label = label.sum(axis=0)
    return label


def load_data():
    image_list = glob.glob(IMAGE_PATH+"*")
    X = []
    y = []
    for image_path in image_list:
        image_name = image_path.split("/")[-1][:-4]
        print("[INFO] Loading on image path: ", image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X.append(image)
        label = load_label(image_name, SAMPLE_LABEL_DICT)
        y.append(label)
    return X, y


X, y = load_data()
