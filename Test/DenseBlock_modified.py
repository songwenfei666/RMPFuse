import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 均值归一化函数
def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)    # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data

# 卷积层，带有reflection padding
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

# 密集块（Dense Block）
first_output_channel=62
class DenseBlock_modified(nn.Module):
    def __init__(self):
        super(DenseBlock_modified, self).__init__()
        self.conv1 = ConvLayer(2, out_channels=first_output_channel, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(64, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = ConvLayer(128, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = ConvLayer(192, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(256, out_channels=64, kernel_size=3, stride=1)

    def forward(self, x1, x2):
        x1 = x1 - 0.5
        x2 = x2 - 0.5
        x = torch.cat([x1, x2], dim=1)

        out1 = self.conv1(x)
        out1 = mean_normalization_tensor(out1)
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        out2 = mean_normalization_tensor(out2)
        out3 = self.conv3(torch.cat([x, out1, out2], dim=1))
        out3 = mean_normalization_tensor(out3)
        out4 = self.conv4(torch.cat([x, out1, out2, out3], dim=1))
        out4 = mean_normalization_tensor(out4)
        out5 = self.conv5(torch.cat([x, out1, out2, out3, out4], dim=1))
        out5 = mean_normalization_tensor(out5)

        return out5

# 测试模型
if __name__ == '__main__':
    input1 = torch.randn(1, 1, 128, 128)
    input2 = torch.randn(1, 1, 128, 128)
    model = DenseBlock_modified()
    output = model(input1, input2)
    print(output.size())
