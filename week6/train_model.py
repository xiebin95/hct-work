# model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D,BatchNormalization,Activation
from tensorflow.keras.layers import Dropout, Input, Add,ZeroPadding2D,Softmax
from tensorflow.keras.initializers import Constant
from week6.lossse import *
from week6.utils.data_feeder import train_image_gen

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


#VGG16基础模型
class VGG16_Basic(tf.keras.Model):

    def __init__(self):
        super(VGG16_Basic,self).__init__()
        #   CONV第一层
        self.Conv1_1 = Conv2D(filters = 64, kernel_size = (3,3),padding="same", activation="relu")
        self.Conv1_2 = Conv2D(filters = 64, kernel_size = (3,3),padding="same", activation="relu")
        self.Maxpool1 = MaxPooling2D(pool_size = (3,3),strides = 2)

        # CONV第二层
        self.Conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        self.Maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=2)

        # CONV第三层
        self.Conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.Maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=2)

        # CONV第四层
        self.Conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=2)

        # CONV第五层
        self.Conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.Maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=2)


    def __call__(self,x):
        f1 =  self.Maxpool1(self.Conv1_2(self.Conv1_1(x)))
        f2 = self.Maxpool2(self.Conv2_2(self.Conv2_1(f1)))
        f3 = self.Maxpool3(self.Conv3_3(self.Conv3_2(self.Conv3_1(f2))))
        f4 = self.Maxpool4((self.Conv4_3(self.Conv4_2(self.Conv4_1(f3)))))
        f5 = self.Maxpool5((self.Conv5_3(self.Conv5_2(self.Conv5_1(f4)))))

        return [f3,f4,f5]

    def net(self,x):
        data_input = Input(shape =x.shape[1:])
        [f3, f4, f5] = self.__call__(data_input)
        net = Model(inputs=data_input, outputs=f5)
        net.build(x.shape)
        net.summary()
        return net



#FCN8S_VGG16
class fcn8s_vgg16(tf.keras.Model):

    def __init__(self,nclass):
        super(fcn8s_vgg16,self).__init__()
        self.encode = VGG16_Basic()

        #全卷积层6
        self.fc6 = Conv2D(filters = 4096, kernel_size = (7,7),padding="same", activation="relu")
        self.drop6 = Dropout(0)

        #全卷积层7
        self.fc7 = Conv2D(filters = 4096, kernel_size = (1,1),padding="same", activation="relu")
        self.drop7 = Dropout(0)

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



    def __call__(self,x,pad =True):
        if pad:
            # x_pad = self.input_pad(x)
            x_pad = ZeroPadding2D((100,100))(x)
        else:
            x_pad = x
        f3, f4, f5 = self.encode(x_pad)
        f6 = self.drop6(self.fc6(f5))
        f7 = self.final_classfier(self.drop7(self.fc7(f6)))

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

    def input_pad(self,x):
        ori_n, ori_h, ori_w, ori_c = x.shape
        x_mask = np.zeros([ori_n, ori_h + 200, ori_w + 200, ori_c]).astype(np.float32)

        x_mask[:, 100:100 + ori_h, 100:100 + ori_w, :] = x[:, :, :, :].astype(np.float32)
        return x_mask

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
        self.nclass = nclass
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
        # # predict = tf.transpose(final_scores, perm=[0, 2, 3, 1])
        # predict = tf.reshape(final_scores, shape=[-1, self.nclass])
        # predict = Softmax()(predict)
        return final_scores

    def net(self,x_shape):
        data_input = Input(shape =x_shape)
        final_scores = self.__call__(data_input)
        print(final_scores)
        net = Model(inputs=data_input, outputs=final_scores)
        # net.build(x.shape)
        # net.summary()
        return net



data_dir = '/home/xb/hct-cv-1/week6/data/train.csv'
train_list = pd.read_csv(data_dir)
# # input_test,mask_test = train_image_gen(train_list)
# print(train_image_gen(train_list))


# print(input_array_label)
# print(input_array_label.shape)

# print(input_array_label)
#fcn8s_resnet101
test_fcn8s_resnet101  = fcn8s_resnet101(8)
# final_scores = test_fcn8s_resnet101(input_array)
# print(final_scores.shape)
print(test_fcn8s_resnet101.net((128,384,3)).summary())
molde_test = test_fcn8s_resnet101.net((128,384,3))
adam = tf.keras.optimizers.Adam()  # 优化函数，设定学习率（lr）等参数
molde_test.compile(loss=categorical_crossentropy_with_logits, optimizer=adam, metrics=['accuracy'])
# molde_test.fit_generator(train_image_gen(train_list),steps_per_epoch = len(train_list)//4, epochs= 100)
molde_test.fit_generator(train_image_gen(train_list),steps_per_epoch = 1, epochs= 1)
molde_test.save('./model')
