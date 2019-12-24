# model
import numpy as np
import cv2
import tensorflow as tf
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

    def __call__(self, input_tensor, n_classes=21, *args, **kwargs):
        x = Conv2D(64, 7, strides=2)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(3, 2)(x)

        x = self._make_layer(x, 64,3)
        f3 = self._make_layer(x, 128,4)
        f4 = self._make_layer(f3, 256,23)
        f5 = self._make_layer(f4, 512,3)

        return f3, f4, f5

    def net(self, x):
        data_input = Input(shape=x.shape[1:])
        f3, f4, f5 = self.__call__(data_input)
        net = Model(inputs=data_input, outputs=f5)
        net.build(x.shape)
        net.summary()
        return net

#fcn8s_resnet101
class fcn8s_resnet101(tf.keras.Model):

    def __init__(self,nclass):
        super(fcn8s_resnet101,self).__init__()
        self.encode = ResNet101()

        #classfier
        self.final_classfier = Conv2D(filters = nclass, kernel_size = (1,1),padding="same")
        self.f3_classfier = Conv2D(filters = nclass, kernel_size = (1,1),padding="same")
        self.f4_classfier = Conv2D(filters = nclass, kernel_size = (1,1),padding="same")

        self.up2time = Conv2DTranspose(filters=nclass,
                                           kernel_size=(4, 4),
                                           strides=(2, 2),
                                           padding='same',
                                           activation='sigmoid',
                                           kernel_initializer=Constant(bilinear_upsample_weights(2, nclass)))
        self.up4time = Conv2DTranspose(filters=nclass,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_initializer=Constant(bilinear_upsample_weights(2, nclass)))

        self.up32time = Conv2DTranspose(filters=nclass,
                                       kernel_size=(16, 16),
                                       strides=(8, 8),
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_initializer=Constant(bilinear_upsample_weights(8, nclass)))



    def __call__(self,x):

        f3, f4, f5 = self.encode(x)

        f7 = self.final_classfier(f5)

        up2_feat = self.up2time(f7)
        h = self.f3_classfier(f4)
        h = h[:, h.shape[1]-up2_feat.shape[1]: h.shape[1], h.shape[2]-up2_feat.shape[2]:h.shape[2],:]
        h = h + up2_feat

        up4_feat = self.up4time(h)
        h = self.f4_classfier(f3)
        h = h[:, h.shape[1]-up4_feat.shape[1]:h.shape[1], h.shape[2]-up4_feat.shape[2]:h.shape[2], :]
        h = h + up4_feat

        h = self.up32time(h)
        final_scores = h[:, h.shape[1]-x.shape[1]:h.shape[1], h.shape[2]-x.shape[2]:h.shape[2], :]

        return final_scores

    def net(self,x):
        data_input = Input(shape =x.shape[1:])
        final_scores = self.__call__(data_input)
        net = Model(inputs=data_input, outputs=final_scores)
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
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = cv2.imread('/home/xb/图片/1.jpeg')
h,w,c = img.shape
print(h,w)
img_result = bilinear_interpolation(img,(w*2,h*2))
print(img_result.shape)
cv2.imwrite('img_result.png',img_result)
