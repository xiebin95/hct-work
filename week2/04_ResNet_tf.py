from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def BasicNet(input_tensor, pre_channel, in_channel, stride=1, *args, **kwargs):
    out_channel = in_channel
    if stride > 1 or pre_channel != out_channel:
        shortcut = Conv2D(out_channel, 1, strides=stride)(input_tensor)
    else:
        shortcut = input_tensor
    x = ZeroPadding2D((1, 1))(input_tensor)
    x = Conv2D(in_channel, 3, strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(out_channel, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def BottleNeck(input_tensor, pre_channel, in_channel, stride=1, *args, **kwargs):
    out_channel = in_channel * 4
    if stride > 1 or pre_channel != out_channel:
        shortcut = Conv2D(out_channel, 1, strides=stride)(input_tensor)
    else:
        shortcut = input_tensor
    x = Conv2D(in_channel, 1, strides=stride)(input_tensor)
    print(x.shape)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(in_channel, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(out_channel, 1)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


class ResNet(object):
    def __call__(self, name, block, input_shape=(224, 224, 3), n_classes=1000, *args, **kwargs):
        input_tensor = Input(shape=input_shape)
        x = ZeroPadding2D((3, 3))(input_tensor)
        x = Conv2D(64, 7, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D(3, 2)(x)

        x = self._make_layer(x, name, block, 2)
        x = self._make_layer(x, name, block, 3)
        x = self._make_layer(x, name, block, 4)
        x = self._make_layer(x, name, block, 5)

        x = GlobalAveragePooling2D()(x)
        x = Dense(n_classes)(x)

        return Model(inputs=input_tensor, outputs=x)

    def _make_layer(self, input_tensor, name, block, stage):
        print(input_tensor.shape)
        name_dict = {
            "resnet18": {"pre_ch": [-1, 64, 64, 128, 256, 512], "in_ch": [64, 64, 128, 256, 512], "layers": [0, 2, 2, 2, 2]},
            "resnet34": {"pre_ch": [-1, 64, 64, 128, 256, 512], "in_ch": [64, 64, 128, 256, 512], "layers": [0, 3, 4, 6, 3]},
            "resnet50": {"pre_ch": [-1, 64, 256, 512, 1024, 2048], "in_ch": [64, 64, 128, 256, 512], "layers": [0, 3, 4, 6, 3]},
            "resnet101": {"pre_ch": [-1, 64, 256, 512, 1024, 2048], "in_ch": [64, 64, 128, 256, 512],
                         "layers": [0, 3, 4, 23, 3]},
            "resnet152": {"pre_ch": [-1, 64, 256, 512, 1024, 2048], "in_ch": [64, 64, 128, 256, 512],
                         "layers": [0, 3, 8, 36, 3]},
        }
        name = name.lower()
        stride = 1 if stage == 2 else 2
        pre_ch_0 = name_dict[name]["pre_ch"][stage - 1]
        in_ch_0 = name_dict[name]["in_ch"][stage - 1]
        pre_ch = name_dict[name]["pre_ch"][stage]
        in_ch = name_dict[name]["in_ch"][stage - 1]
        layer_num = name_dict[name]["layers"][stage-1]
        x = block(input_tensor, pre_ch_0, in_ch_0, stride)
        for _ in range(1, layer_num):
            x = block(x, pre_ch, in_ch)

        return x


def resnet18():
    return ResNet()("resnet18", BasicNet)


def resnet34():
    return ResNet()("resnet34", BasicNet)


def resnet50():
    return ResNet()("resnet50", BottleNeck)


def resnet101():
    return ResNet()("resnet101", BottleNeck)


def resnet152():
    return ResNet()("resnet152", BottleNeck)


if __name__ == '__main__':
    # net = resnet18()
    # net = resnet34()
    # net = resnet50()
    # net = resnet101()
    net = resnet152()
    net.build((1000, 224, 224, 3))
    net.summary()
