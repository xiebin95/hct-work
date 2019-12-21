# model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.initializers import Constant

# from tensorflow.nn import conv2d_transpose

image_shape = (160, 576)


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


class MyModel(tf.keras.Model):
    def __init__(self, n_class):
        super().__init__()
        self.vgg16_model = self.load_vgg()

        self.conv_test = Conv2D(filters=n_class, kernel_size=(1, 1))  # 分类层
        self.deconv_test = Conv2DTranspose(filters=n_class,
                                           kernel_size=(64, 64),
                                           strides=(32, 32),
                                           padding='same',
                                           activation='sigmoid',
                                           kernel_initializer=Constant(bilinear_upsample_weights(32, n_class)))  # 上采样层



    def load_vgg(self):
        # 加载vgg16模型，其中注意input_tensor，include_top
        vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                                        input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))
        for layer in vgg16_model.layers[:15]:
            layer.trainable = False  # 不训练前15层模型
        return vgg16_model

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




    def forward(self,x):
        f1 =  self.Maxpool1(self.Conv1_2(self.Conv1_1(x)))
        f2 = self.Maxpool2(self.Conv2_2(self.Conv2_1(f1)))
        f3 = self.Maxpool3(self.Conv3_3(self.Conv3_2(self.Conv3_1(f2))))
        f4 = self.Maxpool4((self.Conv4_3(self.Conv4_2(self.Conv4_1(f3)))))
        f5 = self.Maxpool5((self.Conv5_3(self.Conv5_2(self.Conv5_1(f4)))))
        return f3,f4,f5
test  = VGG16_Basic()
input_array = tf.random.truncated_normal([10,224,224,3],mean=0,stddev=1)
f3,f4,f5 = test.forward(input_array)
print(f3.shape,f4.shape,f5.shape)