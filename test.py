import keras
import cv2
import numpy as np
import model
from keras.utils.generic_utils import CustomObjectScope
COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 150, 0), (255, 255, 255),
         (150, 150, 150), (0, 255, 100), (0, 0, 0)]
imaeg_name = "image1.jpg"
with CustomObjectScope({'relu6': model.relu6}):
    model = keras.models.load_model("./sample.h5")
print("loaded model")
image = cv2.imread("./dataset/Images/" + imaeg_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array([image])
output = model.predict(image)[0]
output= np.argmax(output,axis=1)
mask= np.zeros(shape=(output.shape[0],3))

mask[output==0]=COLOR[0]
mask[output==1]=COLOR[1]
mask[output==2]=COLOR[2]
mask[output==3]=COLOR[3]
mask[output==4]=COLOR[4]
mask[output==5]=COLOR[5]
mask[output==6]=COLOR[6]
mask[output==7]=COLOR[7]
mask= np.reshape(mask,(720,1280,3))
cv2.imshow("segment",mask)
cv2.imshow("image", image[0])
cv2.waitKey(0)