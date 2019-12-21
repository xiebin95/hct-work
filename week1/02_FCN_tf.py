import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Input
from tensorflow.keras import Model
# from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Input
# from keras import Model


class VGG16(object):
    def base_net(self, n_classes):
        # conv1
        data_input = Input(shape=(224, 224, 3))
        conv1_1 = Conv2D(64, 3, padding="same", activation="relu")(data_input)
        conv1_2 = Conv2D(64, 3, padding="same", activation="relu")(conv1_1)
        max_pool1 = MaxPooling2D()(conv1_2)
        # conv2
        conv2_1 = Conv2D(128, 3, padding="same", activation="relu")(max_pool1)
        conv2_2 = Conv2D(128, 3, padding="same", activation="relu")(conv2_1)
        max_pool2 = MaxPooling2D()(conv2_2)
        # conv3
        conv3_1 = Conv2D(256, 3, padding="same", activation="relu")(max_pool2)
        conv3_2 = Conv2D(256, 3, padding="same", activation="relu")(conv3_1)
        conv3_3 = Conv2D(256, 3, padding="same", activation="relu")(conv3_2)
        max_pool3 = MaxPooling2D()(conv3_3)
        # conv4
        conv4_1 = Conv2D(512, 3, padding="same", activation="relu")(max_pool3)
        conv4_2 = Conv2D(512, 3, padding="same", activation="relu")(conv4_1)
        conv4_3 = Conv2D(512, 3, padding="same", activation="relu")(conv4_2)
        max_pool4 = MaxPooling2D()(conv4_3)
        # conv5
        conv5_1 = Conv2D(512, 3, padding="same", activation="relu")(max_pool4)
        conv5_2 = Conv2D(512, 3, padding="same", activation="relu")(conv5_1)
        conv5_3 = Conv2D(512, 3, padding="same", activation="relu")(conv5_2)
        max_pool5 = MaxPooling2D()(conv5_3)
        # fc6--->full conv6
        fc6 = Conv2D(4096, 3,padding="same", activation="relu")(max_pool5)
        drop6 = Dropout(0.5)(fc6)
        # fc7--->full conv7
        fc7 = Conv2D(4096, 1, activation="relu")(drop6)
        drop7 = Dropout(0.5)(fc7)
        # fc8--->full conv8
        fc8 = Conv2D(n_classes, 1)(drop7)

        return data_input, max_pool3, max_pool4, fc8


class VGG16_FCN32s(VGG16):
    def net(self, n_classes):
        data_in, pool3, pool4, fc8 = self.base_net(n_classes)
        up_sample32 = UpSampling2D(size=(32, 32))(fc8)
        net = Model(inputs=data_in, outputs=up_sample32)
        net.compile("adam", "mse")
        net.build((None, 224, 224, 3))
        net.summary()
        return net


if __name__ == '__main__':
    # data = np.random.standard_normal([1, 224, 224, 3])
    print(tf.__version__)
    n = VGG16_FCN32s().net(21)
