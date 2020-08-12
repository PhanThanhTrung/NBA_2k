from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

IMAGE_ORDERING = "channels_last"
BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')
MERGE_AXIS = -1


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1),
                      name='conv1_pad',
                      data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(filters,
               kernel,
               data_format=IMAGE_ORDERING,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D((1, 1),
                      data_format=IMAGE_ORDERING,
                      name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                        data_format=IMAGE_ORDERING,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               data_format=IMAGE_ORDERING,
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def get_mobilenet_encoder(input_height=224,
                          input_width=224,
                          pretrained='imagenet'):

    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3

    img_input = Input(shape=(input_height, input_width, 3))

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x

    x = _depthwise_conv_block(x,
                              128,
                              alpha,
                              depth_multiplier,
                              strides=(2, 2),
                              block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x

    x = _depthwise_conv_block(x,
                              256,
                              alpha,
                              depth_multiplier,
                              strides=(2, 2),
                              block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x

    x = _depthwise_conv_block(x,
                              512,
                              alpha,
                              depth_multiplier,
                              strides=(2, 2),
                              block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x

    x = _depthwise_conv_block(x,
                              1024,
                              alpha,
                              depth_multiplier,
                              strides=(2, 2),
                              block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x

    if pretrained == 'imagenet':
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(model_name, weight_path)

        Model(img_input, x).load_weights(weights_path)

    if pretrained == 'imagenet':
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(model_name, weight_path)

        Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5]


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
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3),
                padding='valid',
                activation='relu',
                data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    return model


def mobilenet_unet(n_classes,
                   input_height=224,
                   input_width=224,
                   encoder_level=3):

    model = _unet(n_classes,
                  get_mobilenet_encoder,
                  input_height=input_height,
                  input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == "__main__":
    model = mobilenet_unet(8, 1280, 720, 3)
    model.summary()