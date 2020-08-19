import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from models.mobilenet import get_mobilenet_encoder
from models.resnet import get_resnet50_encoder
IMAGE_ORDERING = "channels_last"
MERGE_AXIS = -1  #if image ordering is channels last else 1


def _unet(n_classes,
          encoder,
          l1_skip_conn=True,
          input_height=416,
          input_width=608):

    img_input, levels = encoder(input_height=input_height,
                                input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    f3 = ZeroPadding2D(padding=((1, 0),(0, 0)), data_format=IMAGE_ORDERING)(f3) #zeropadding to make f3 has the same shape as o
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    f2= ZeroPadding2D(padding=((0, 2),(0, 0)), data_format=IMAGE_ORDERING)(f2) #zeropadding to make f2 has the same shape as o
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        f1= ZeroPadding2D(padding=((2, 2),(0, 0)), data_format=IMAGE_ORDERING)(f1) #zeropadding to make f1 has the same shape as o
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)  #addition
    o = Cropping2D(cropping=((2,2),(0,0)),data_format=IMAGE_ORDERING)(o)
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224):

    model = _unet(n_classes,
                  get_mobilenet_encoder,
                  input_height=input_height,
                  input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608):

    model = _unet(n_classes,
                  get_resnet50_encoder,
                  input_height=input_height,
                  input_width=input_width)
    model.model_name = "resnet50_unet"
    return model


if __name__ == "__main__":
    model = mobilenet_unet(8, 360, 640)
    print(model.summary())