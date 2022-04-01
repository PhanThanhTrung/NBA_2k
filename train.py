import os
import keras
from models.unet import mobilenet_unet
from Data_processing.data_preprocessing import *

model = mobilenet_unet(8, 360, 640)
model.summary()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])
model.fit_generator(batch_generator(batch_size=4, augment=False, output_height=360),
                    steps_per_epoch=20,
                    epochs=40,
                    verbose=True)
if os.path.exists("./Model_weight/") == False:
    os.makedirs("./Model_weight/")
model.save_weights("./Model_weight/mobile_net_360_640.h5")
