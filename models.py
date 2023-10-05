import torch.nn as nn
import torch.nn.functional as F

class resBlock(nn.Module):
    def __init__(self, in_channel):
        super(resBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),  # 保证conv后shape(w,h)不变(3*3卷积padding为1即可)
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.InstanceNorm2d(in_channel),  # 沿着通道归一化
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),  # 保证conv后shape不变
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.InstanceNorm2d(in_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)  # 把这几个小单元使用Sequential封装起来

    def forward(self, x):
        return x + self.conv_block(x)   #以上定义了一个resnet基本单元

class Generator(nn.Module):   # 生成器， 先下采样再上采样
    def __init__(self):
        super(Generator, self).__init__()
        # 定义网络核心
        net = [
            nn.ReflectionPad2d(3),  # 保证conv后shape不变
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # 定义下采样模块
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel * 2
        # ResBlock
        for _ in range(9):
            net += [resBlock(in_channel)]

        # 定义上采样模块
        out_channel = in_channel // 2
        for _ in range(2):
            net += [nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel, \
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                    ]
            in_channel = out_channel
            out_channel = in_channel // 2
        # 输出层
        net += [
            nn.ReflectionPad2d(3),  # 保证conv后shape不变
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 主要为简单的下采样
        model = [nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(* model)

    def forward(self, x):
        x = self.model(x)
        #把feature map大小弄成1*1
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# 测试一下

# if __name__=='__main__':
#     G = Generator()
#     D = Discriminator()
#
#     import torch
#     input_tensor = torch.ones((1, 3, 256, 256), dtype=torch.float)
#     out = G(input_tensor)
#     print(out.size())
#
#     out = D(input_tensor)
#     print(out.size())