import torch
from torch import nn


def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)  # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ReLU = nn.ReLU()

        # 路线1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=64, out_channels=63, kernel_size=1)


        # 路线2，1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)


        # 路线3，1×1卷积层, 7×7的卷积
        self.p3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=3)


        # 路线4，3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)

        self.p5_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.p5_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)


        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)

    def forward(self, x):
        t = x - 0.5
        x = self.conv1(x)
        x = mean_normalization_tensor(x)
        x = self.ReLU(x)

        # 路线1
        p1 = self.p1_1(x)
        p1 = mean_normalization_tensor(p1)
        p1 = self.ReLU(p1)


        # 路线2
        p2 = self.p2_1(x)
        p2 = mean_normalization_tensor(p2)
        p2 = self.ReLU(p2)

        # 路线3
        p3 = self.p3_1(x)
        p3 = mean_normalization_tensor(p3)
        p3 = self.ReLU(p3)

        # 路线4
        p4 = self.p4_1(x)
        p4 = self.p4_2(p4)
        p4 = mean_normalization_tensor(p4)
        p4 = self.ReLU(p4)

        #线路5
        p5 = self.p5_1(x)
        p5 = self.p5_2(p5)
        p5 = mean_normalization_tensor(p5)
        p5 = self.ReLU(p5)


        out = torch.cat((t, p1, p2, p3, p4, p5), dim=1)
        out = self.conv2(out)
        out = mean_normalization_tensor(out)
        out = self.ReLU(out)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 640, 480)
    model = InceptionWithCBAM()
    y = model(x)
    print(y.shape)
