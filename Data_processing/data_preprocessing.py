import cv2
import glob
import imutils
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from Data_processing.utils import shuffle, load_mask, load_label

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

aug = ImageDataGenerator(rescale=1. / 255,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True)


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


def batch_generator(batch_size=4, augment=True, output_height=720):
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
                image_name = image_path.split("/")[-1][:-4]
                label = load_mask(image_name)
                if label.shape[0] != output_height:
                    image = imutils.resize(img, height=output_height)    
                    label = imutils.resize(label, height=int(output_height/2))
                X_train.append(image)
                y_train.append(label)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            y_train = to_categorical(y_train, len(SAMPLE_LABEL_DICT))
            if augment:
                X = aug.flow(X_train, y_train, batch_size=batch_size, seed=1)
                X_train, y_train = next(X)
            yield X_train, y_train


if __name__ == '__main__':
    load_mask("image1")
    #X, y = load_data()
