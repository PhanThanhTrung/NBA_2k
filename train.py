import numpy as np
import keras
from model import mobilenet_unet
from data_preprocessing import *

model = mobilenet_unet(8, 720, 1280, 3)
model.summary()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])
model.fit_generator(batch_generator(batch_size=8),
                    steps_per_epoch=4,
                    epochs=10,
                    verbose=True)
