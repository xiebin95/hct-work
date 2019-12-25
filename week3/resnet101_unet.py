# model
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import  concatenate
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D,BatchNormalization,Activation
from tensorflow.keras.layers import Dropout, Input, Add,ZeroPadding2D
from tensorflow.keras.initializers import Constant


#上采样模块权重初始化
def bilinear_upsample_weights(factor, number_of_classes):
    """初始化权重参数"""


    filter_size = factor * 2 - factor % 2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights




#ResNet101基础结构
class ResNet101(tf.keras.Model):

    def BottleNeck(self,input_tensor, in_channel, stride=1):
        out_channel = in_channel * 4
        if stride > 1:
            if in_channel == 64:
                shortcut = Conv2D(out_channel, 1, strides=1)(input_tensor)
                stride = 1
            else:
                shortcut = Conv2D(out_channel, 1, strides=stride)(input_tensor)
        else:
            shortcut = input_tensor
        x = Conv2D(in_channel, 1, strides=stride)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(in_channel, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(out_channel, 1)(x)
        x = BatchNormalization()(x)
        x += shortcut
        x = Activation("relu")(x)
        return x



    def _make_layer(self, input_tensor, kernel_num, layer_num):
        x = self.BottleNeck(input_tensor, kernel_num, 2)
        for _ in range(1, layer_num):
            x = self.BottleNeck(x, kernel_num, 1)

        return x

    def __call__(self, input_tensor):
        x = Conv2D(64, 7, strides=2)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(3, 2)(x)

        f2 = self._make_layer(x, 64,3)
        f3 = self._make_layer(f2, 128,4)
        f4 = self._make_layer(f3, 256,23)
        f5 = self._make_layer(f4, 512,3)

        return f2,f3, f4, f5

    def net(self, x):
        data_input = Input(shape=x.shape[1:])
        f2,f3, f4, f5 = self.__call__(data_input)
        net = Model(inputs=data_input, outputs=f5)
        net.build(x.shape)
        net.summary()
        return net




def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

class UNetConvBlock(tf.keras.Model):
    def __call__(self, input_tensor, out_chans):
        super(UNetConvBlock, self).__init__()
        #第一次卷积
        out = Conv2D(filters=out_chans, kernel_size=(3, 3), padding="valid")(input_tensor)
        out = Activation("relu")(out)
        out = BatchNormalization()(out)

        #第二次卷积
        out = Conv2D(filters=out_chans, kernel_size=(3, 3), padding="valid")(out)
        out = Activation("relu")(out)
        out = BatchNormalization()(out)
        return out

class UNetUpBlock(tf.keras.Model):
    def __init__(self, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        self.outchanels = out_chans
        self.up_mode = up_mode
        if up_mode == 'upconv':
            self.up = Conv2DTranspose(filters=out_chans,
                                           kernel_size=(4, 4),
                                           strides=(2, 2),
                                           padding='same',
                                           activation='sigmoid')
        elif up_mode == 'upsample':
            self.up = UpSampling2D(size = 2,interpolation = 'bilinear')
            # self.up = UpSampling2D()


        self.conv_block = UNetConvBlock()

    def center_crop(self, layer, target_size):
        _, layer_height, layer_width,_ = layer.shape

        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :,  diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1]),:
        ]

    def __call__(self, x, bridge):
        if self.up_mode == 'upsample':
            up =  Conv2D(filters=self.outchanels, kernel_size=(1, 1))(self.up(x))
        else:
            up = self.up(x)

        crop1 = self.center_crop(bridge, up.shape[1:3])
        print(up.shape,crop1.shape)
        out = concatenate((up, crop1), -1)
        print(out.shape)
        out = self.conv_block(out,up.shape[-1])


        return out

#unet_resnet101
class unet_resnet101(tf.keras.Model):

    def __init__(
            self,
            n_classes=2,
            depth=5,
            wf=6,
            up_mode='upconv',
    ):
        super(unet_resnet101, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.depth = depth
        self.encode = ResNet101()
        self.up_path = []
        for i in reversed(range(2, depth)):
            self.up_path.append(
                UNetUpBlock( 2 ** (wf + i), up_mode)
            )

        self.last = Conv2D(filters=n_classes, kernel_size=(1, 1))

    def __call__(self, x):
        f2,f3,f4,f5 = self.encode(x)
        blocks = [ f2,f3,f4,f5]
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            print(i)
            print(x.shape)
            x = up(x, blocks[-i - 2])
            print(x.shape)

        return self.last(x)
input_array = tf.random.truncated_normal([10,572,572,1],mean=0,stddev=1)
unet_resnet101_test = unet_resnet101()
result = unet_resnet101_test(input_array)
print(result.shape)