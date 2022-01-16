import torch.nn as nn
import torch.nn.functional as F

from ..utils import Conv_BN_ReLU

class FPEM_v1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FPEM_v1, self).__init__()
        planes=out_channels
        self.dwconv3_1=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=planes,
                                 bias=False)

        self.smooth_layer3_1=Conv_BN_ReLU(planes,planes)

        self.dwconv2_1=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=planes,
                                 bias=False
                                 )

        self.smooth_layer2_1=Conv_BN_ReLU(planes,planes)

        self.dwconv1_1=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=planes,
                                 bias=False)

        self.smooth_layer1_1=Conv_BN_ReLU(planes,planes)

        self.dwconv2_2=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 groups=planes,
                                 bias=False)

        self.smooth_layer2_2=Conv_BN_ReLU(planes,planes)

        self.dwconv3_2=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 groups=planes,
                                 bias=False)

        self.smooth_layer3_2=Conv_BN_ReLU(planes,planes)

        self.dwconv4_2=nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 groups=planes,
                                 bias=False)

        self.smooth_layer4_2=Conv_BN_ReLU(planes,planes)


    def _upsample_add(self,x,y):
        _,_,H,W=y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')+y

    def forward(self,f1,f2,f3,f4):
        #torch.Size([1, 128, 160, 160])
        #torch.Size([1, 128, 80, 80])
        #torch.Size([1, 128, 40, 40])
        #torch.Size([1, 128, 20, 20])

        #self._upsample_add(f4,f3).shape:   torch.Size([1, 128, 40, 40])
        #torch.Size([1, 128, 40, 40])不变

        #torch.Size([1, 128, 40, 40])
        f3=self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4,f3)))
        #torch.Size([1, 128, 80, 80])
        f2=self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3,f2)))
        #torch.Size([1, 128, 160, 160])
        f1=self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2,f1)))

        #print(self._upsample_add(f2,f1).shape)torch.Size([1, 128, 160, 160])
        f2=self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2,f1)))
        #torch.Size([1, 128, 80, 80])
        f3=self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3,f2)))
        #torch.Size([1, 128, 40, 40])
        f4=self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4,f3)))
        #torch.Size([1, 128, 20, 20])

        return f1,f2,f3,f4
        # torch.Size([1, 128, 160, 160])
        # torch.Size([1, 128, 80, 80])
        # torch.Size([1, 128, 40, 40])
        # torch.Size([1, 128, 20, 20])
    #与FPN的不同，FPN是单独up_scale Enhancement模块
    #fpem再加上down_scale Enhancement模块
    #总体所有通道并没有改变，但随着nc的增加，不同尺度的特征图融合更加充分，特征的感受域变大，增强了感受域










