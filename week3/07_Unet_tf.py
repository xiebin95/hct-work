from keras.layers import Input, Conv2D, MaxPooling2D, Cropping2D, Activation, UpSampling2D, Add, BatchNormalization
from keras.models import Model


class Unet(object):
    def __init__(self, depth, n_classes=2, input_shape=(572, 572, 1)):
        self.input_shape = input_shape
        self.depth = depth
        self.n_classes = n_classes

    def _down_sample_block(self, x, deep, out_channel):
        """下采样块(encoder)，max_pool, conv, conv"""
        if deep > 0:
            x = MaxPooling2D()(x)
        x = BatchNormalization()(x)
        x = Conv2D(out_channel, 3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(out_channel, 3, activation="relu")(x)
        return x

    def _up_sample_block(self, x, shortcut, out_channel):
        """
        上采样块(decoder), 先上采样，
        再把shortcut进行crop、调整维度,
        再Add, conv, conv
        :param x:
        :param shortcut:
        :param out_channel:
        :return:
        """
        x = UpSampling2D()(x)
        channel = x.shape[3]

        crop1 = shortcut.shape[1] - x.shape[1]
        crop1_1 = crop1 // 2
        crop1_2 = crop1 - crop1_1
        crop2 = shortcut.shape[2] - x.shape[2]
        crop2_1 = crop2 // 2
        crop2_2 = crop2 - crop2_1
        shortcut = Cropping2D(((crop1_1, crop1_2), (crop2_1, crop2_2)))(shortcut)
        shortcut = Conv2D(channel, 1)(shortcut)

        x = Add()([shortcut, x])
        x = BatchNormalization()(x)
        x = Conv2D(out_channel, 3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(out_channel, 3, activation="relu")(x)
        return x

    def build_network(self):
        """encoder + decoder"""
        input_tensor = Input(shape=self.input_shape)
        out = input_tensor
        out_channel = 64
        shortcuts = []
        for d in range(self.depth):
            out = self._down_sample_block(out, d, out_channel)
            shortcuts.append(out)
            out_channel *= 2

        out_channel //= 2
        for u in range(self.depth - 2, -1, -1):
            out_channel //= 2
            out = self._up_sample_block(out, shortcuts[u], out_channel)

        out = Conv2D(self.n_classes, 1)(out)
        out = Activation("softmax")(out)
        return Model(inputs=input_tensor, outputs=out)


if __name__ == '__main__':
    net = Unet(5).build_network()
    net.summary()
