import torch,cv2
import  numpy as np
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    center = kernel_size / 2
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

a = Block(3, 64, 7, 3, 2)
b = torch.randn((1,3, 256,256))
print(a(b))
print(a.forward(b))
print(a(b) == a.forward(b) )
img = cv2.imread("/home/xb/hct-cv/week2/2-2课程资料/微信图片_20191207203405.jpg")
print(img.shape)
x = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).unsqueeze(0)
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
# 将其定义为 bilinear kernel
conv_trans.weight.data = bilinear_kernel(3, 3, 4)
y = conv_trans(x).data.squeeze().permute(1, 2, 0).numpy()
cv2.imshow('y',y.astype('uint8'))
cv2.imshow('img',img)
cv2.waitKey()
print(y.shape)