import keras
import cv2
from imutils import resize
import numpy as np
from models.unet import *
import time
from Data_processing.utils import process_mask, load_mask
MODEL_PATH = "/Users/hit.flouxetine/Desktop/NBA_2k/Model_weight/mobile_net_360_640.h5"
IMAGE_PATH = "/Users/hit.flouxetine/Desktop/NBA_2k/dataset/Images/"
COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 150, 0), (255, 255, 255),
         (150, 150, 150), (0, 255, 100), (0, 0, 0)]
image_name = "image1.png"
model = mobilenet_unet(8, 360, 640)
model.load_weights(MODEL_PATH)
print("loaded model")
t=time.time()
image = cv2.imread(IMAGE_PATH + image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = resize(image, height=360)
image = np.array([image])
output = model.predict(image)[0]
output = np.argmax(output, axis=2)
print(time.time()-t)
mask1 = process_mask(output, COLOR, shape=(180,320,3))
#mask1 = resize(mask1, height=720)
mask2 = process_mask(load_mask(image_name[:-4]), COLOR)
cv2.imshow("segment", mask1)
cv2.imshow("ground truth", mask2)
cv2.imshow("image", image[0])
cv2.waitKey(0)