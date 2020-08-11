import numpy as np
import keras
from model import mobilenet_unet
from data_preprocessing import load_data
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#load, and prepairing data for training
X,y=load_data()
X=np.array(X)
y=np.array(y)
y=np.reshape(y,(-1,y.shape[1]*y.shape[2]))
y=to_categorical(y,num_classes=8)

#split data to test_set, and train_set. Apply augmentation to train set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

#define model
model=mobilenet_unet(8,720,1280,3)
model.summary()

#define optimizer and compile 
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy",optimizer=adam, metrics=["accuracy"])

#training model
model.fit(
	x=aug.flow(X_train, y_train, batch_size=4),
	validation_data=(X_test, y_test),
	epochs=5, verbose=True)
model.save("./sample.h5")