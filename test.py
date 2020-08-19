import keras
import cv2
import numpy as np
from models.unet import *
from Data_processing.utils import process_mask, load_mask
MODEL_PATH = "/Users/hit.flouxetine/Desktop/NBA_2k/Model_weight/model_weight.h5"
IMAGE_PATH = "/Users/hit.flouxetine/Desktop/NBA_2k/dataset/Images/"
COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 150, 0), (255, 255, 255),
         (150, 150, 150), (0, 255, 100), (0, 0, 0)]
image_name = "image1.png"
model = mobilenet_unet(8, 720, 1280)
model.load_weights(MODEL_PATH)
print("loaded model")
image = cv2.imread(IMAGE_PATH + image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array([image])
output = model.predict(image)[0]
output = np.argmax(output, axis=2)
mask1 = process_mask(output, COLOR)
mask2 = process_mask(load_mask(image_name[:-4]), COLOR)
cv2.imshow("segment", mask1)
cv2.imshow("ground truth", mask2)
cv2.imshow("image", image[0])
cv2.waitKey(0)