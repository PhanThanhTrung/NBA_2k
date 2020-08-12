import numpy as np
import keras
from model import mobilenet_unet
from data_preprocessing import *

model = mobilenet_unet(8, 720/2, 1280/2, 3)
model.summary()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])
model.fit_generator(batch_generator(batch_size=4),
                    steps_per_epoch=20,
                    epochs=40,
                    verbose=True)

model.save_weights("./model.h5")
