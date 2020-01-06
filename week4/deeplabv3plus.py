# model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D,ReLU

from tensorflow.keras import backend as K

#moblienetv2基础结构
class MoblieNetv2(tf.keras.Model):
    def block(self,inputs, out_chans, k, s):
        x = Conv2D(out_chans, k, padding='same', strides=s)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6.0)(x)
        return x

    def inverted_residual_block(self,inputs, out_chans, k, t, s, r=False):
        # Depth放大
        tchannel = K.int_shape(inputs)[-1] * t

        x = self.block(inputs, tchannel, 1, 1)
        # 注意这里depth_multiplier这个参数的含义。
        x = DepthwiseConv2D(k, strides=s, depth_multiplier=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6.0)(x)

        x = Conv2D(out_chans, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        if r:
            x = Add()([x, inputs])

        return x

    def make_layer(self,inputs, out_chans, k, t, s, n):
        x = self.inverted_residual_block(inputs, out_chans, k, t, s)

        for i in range(1, n):
            x = self.inverted_residual_block(x, out_chans, k, t, 1, True)

        return x

    def __call__(self, input_tensor):
        x = self.block(input_tensor, 32, (3, 3), s=(2, 2))
        f2 = x
        x = self.inverted_residual_block(x, 16, (3, 3), t=1, s=1)
        x = self.make_layer(x, 24, (3, 3), t=6, s=2, n=2)
        f3 = x
        x = self.make_layer(x, 32, (3, 3), t=6, s=2, n=3)
        f4 = x
        x = self.make_layer(x, 64, (3, 3), t=6, s=2, n=4)
        x = self.make_layer(x, 96, (3, 3), t=6, s=1, n=3)
        f5 = x
        return f2,f3, f4, f5

    def net(self, x):
        data_input = Input(shape=x.shape[1:])
        f2,f3, f4, f5 = self.__call__(data_input)
        net = Model(inputs=data_input, outputs=f5)
        net.build(x.shape)
        net.summary()
        return net
#ASPP
def aspp(inputs):
    depth = inputs.shape[-1]
    #空洞卷积率[1,6,12,18]
    atrous_rates = [6,12,18]
    conv_1x1 = Conv2D( depth, 1, strides=1)(inputs)
    conv_3x3_1 = Conv2D( depth,  3, strides=1, dilation_rate=atrous_rates[0], padding = 'same')(inputs)
    conv_3x3_2 = Conv2D( depth,  3, strides=1, dilation_rate=atrous_rates[1], padding = 'same')(inputs)
    conv_3x3_3 = Conv2D( depth, 3, strides=1, dilation_rate=atrous_rates[2], padding = 'same')(inputs)
    #全局平均
    averagepooling = GlobalAveragePooling2D()(inputs)
    averagepooling = K.reshape(averagepooling,(inputs.shape[0],1,1,inputs.shape[-1]))
    averagepooling_conv_1x1 = Conv2D(depth, 1, strides=1)(averagepooling)
    averagepooling_resize = K.resize_images(averagepooling_conv_1x1,inputs.shape[1],inputs.shape[2],data_format = 'channels_last',interpolation = 'bilinear')

    return  K.concatenate((conv_1x1, conv_3x3_1,conv_3x3_2,conv_3x3_3,averagepooling_resize), -1)

def encode(input_tensor):

    encode_aspp = aspp(input_tensor)
    encode_change_channel = Conv2D(1, 1, 1, padding='same')(encode_aspp)
    encode_upsampling = UpSampling2D((4,4))(encode_change_channel)
    return encode_upsampling


#deeplabv3+
class deeplabv3_plus(tf.keras.Model):


    def __call__(self, input_tensor,nclass):
        base_moblienetv2 = MoblieNetv2()
        f2, f3, f4, f5 = base_moblienetv2(input_tensor)
        #encode
        encode_upsampling = encode(f5)

        #decode
        decode_conv_1x1  = Conv2D(1,1,1,padding = 'same')(f3)
        decode_concatenate =  K.concatenate((decode_conv_1x1,encode_upsampling), -1)
        decode_conv_3x3 = Conv2D(nclass,3,1,padding = 'same')(decode_concatenate)
        decode_upsampling = UpSampling2D((4,4))(decode_conv_3x3)

        return decode_upsampling


deeplabv3_plus_test = deeplabv3_plus()
input_array = tf.random.truncated_normal([10, 224, 224, 3], mean=0, stddev=1)
result_test = deeplabv3_plus_test(input_array,8)
print(result_test)