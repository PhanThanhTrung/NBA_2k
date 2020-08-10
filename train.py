from model import mobilenet_unet
from data_preprocessing import load_data
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

X,y=load_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

model=mobilenet_unet(8,720,1280,3)
model.summary()
H = model.fit(
	x=aug.flow(X_train, y_train, batch_size=4),
	validation_data=(X_test, y_test),
	epochs=5)