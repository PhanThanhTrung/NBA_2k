import os
import glob
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from utils import shuffle, load_label

IMAGE_PATH = './dataset/Images/'
LABEL_PATH = './dataset/Labels/'
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

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

def load_mask()
def load_data():
    image_list = glob.glob(IMAGE_PATH + "*")
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


def batch_generator(batch_size=4):
    image_list = glob.glob(IMAGE_PATH + "*")
    n = len(image_list)
    while True:
        shuffle(image_list)
        for offset in range(0, n, batch_size):
            X_train = []
            y_train = []
            batch_sample = image_list[offset:min(offset + batch_size, n)]
            for image_path in batch_sample:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_train.append(img)
                image_name = image_path.split("/")[-1][:-4]
                label = load_label(image_name, SAMPLE_LABEL_DICT)
                y_train.append(label)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            y_train = y_train.reshape(
                (-1, y_train.shape[1] * y_train.shape[2]))
            y_train = to_categorical(y_train, len(SAMPLE_LABEL_DICT))

            x = aug.flow(X_train, y_train, batch_size=batch_size)
            X_batch, y_batch = next(x)
            yield X_batch, y_batch


if __name__ == '__main__':
    X, y = load_data()
