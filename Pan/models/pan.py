import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU

class PAN(nn.Module):
    def __init__(self,backbone,neck,detection_head):
        super(PAN, self).__init__()
        self.backbone=build_backbone(backbone)

        in_channles=neck.in_channels
        self.reduce_layer1=Conv_BN_ReLU(in_channles[0],128)
        self.reduce_layer2=Conv_BN_ReLU(in_channles[1],128)
        self.reduce_layer3=Conv_BN_ReLU(in_channles[2],128)
        self.reduce_layer4=Conv_BN_ReLU(in_channles[3],128)

        self.fpem1=build_neck(neck)
        self.fpem2=build_neck(neck)

        self.det_head=build_head(detection_head)

    def _upsample(self,x,size,sclae=1):
        _,_,H,W=size
        return F.upsample(x,size=(H//sclae,W//sclae),mode='bilinear')
    ##################################传入值(**data)
    ####################data
    # print(data['imgs'].shape)torch.Size([1, 3, 640, 640])
    # print(data['gt_texts'].shape)torch.Size([1, 640, 640])
    # print(data['gt_kernels'].shape)torch.Size([1, 1, 640, 640])
    # print(data['training_masks'].shape)torch.Size([1, 640, 640])
    # print(data['gt_instances'].shape)torch.Size([1, 640, 640])
    # print(data['gt_bboxes'].shape)torch.Size([1, 201, 4])
    ###################
    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):

        outputs=dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start=time.time()

        #imgs:  [1,3,640,640]
        #backbone
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            if self.training is True:
                gt_texts=gt_texts.cuda()
                gt_kernels=gt_kernels.cuda()
                training_masks=training_masks.cuda()
                gt_instances=gt_instances.cuda()
                gt_bboxes=gt_bboxes.cuda()


        f=self.backbone(imgs)
        #f[0]:  torch.Size([1, 64, 160, 160])
        #f[1]:  torch.Size([1, 128, 80, 80])
        #f[2]:  torch.Size([1, 256, 40, 40])
        #f[3]:  torch.Size([1, 512, 20, 20])直接拿resnet18做下采样

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(
                dict(backbone_time=time.time()-start)
            )

        #reduce channel,将通道数全部变为128
        f1=self.reduce_layer1(f[0])
        #f1:    torch.Size([1, 128, 160, 160])
        f2=self.reduce_layer2(f[1])
        #f2:    torch.Size([1, 128, 80, 80])
        f3=self.reduce_layer3(f[2])
        #f3:    torch.size([1, 128, 40, 40])
        f4=self.reduce_layer4(f[3])
        #f4:    torch.Size([1, 128, 20, 20])

        #FPEM
        f1_1,f2_1,f3_1,f4_1=self.fpem1(f1,f2,f3,f4)
        #print(f1_1.shape)torch.Size([1, 128, 160, 160])
        #print(f2_1.shape)torch.Size([1, 128, 80, 80])
        #print(f3_1.shape)torch.Size([1, 128, 40, 40])
        #print(f4_1.shape)torch.Size([1, 128, 20, 20])

        f1_2,f2_2,f3_2,f4_2=self.fpem2(f1_1,f2_1,f3_1,f4_1)
        #print(f1_2.shape)torch.Size([1, 128, 160, 160])
        #print(f2_2.shape)torch.Size([1, 128, 80, 80])
        #print(f3_2.shape)torch.Size([1, 128, 40, 40])
        #print(f4_2.shape)torch.Size([1, 128, 20, 20])

        #FFM fuse the feature pyramids of different depth
        #两者的底层和高层语义信息对于语义分割非常重要
        #组合这些金字塔最有效的方法是上采样和连接(一般法)，但是这样的后果是nc非常大 (128*4*nc)
        #先连接，再上采样
        f1=f1_1+f1_2#torch.Size([1, 128, 160, 160])
        f2=f2_1+f2_2#torch.Size([1, 128, 80, 80])
        f3=f3_1+f3_2#torch.Size([1, 128, 40, 40])
        f4=f4_1+f4_2#torch.Size([1, 128, 20, 20])

        f2=self._upsample(f2,f1.size())
        #torch.Size([1, 128, 160, 160])
        f3=self._upsample(f3,f1.size())
        #torch.Size([1, 128, 160, 160])
        f4=self._upsample(f4,f1.size())
        # torch.Size([1, 128, 160, 160])

        f=torch.cat((f1,f2,f3,f4),1)#torch.Size([1, 512, 160, 160]) 128*4

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(
                dict(neck_time=time.time()-start)
            )
            start=time.time()

        #detection
        #torch.Size([1, 512, 160, 160])
        det_out=self.det_head(f)
        #torch.Size([1, 6, 160, 160])

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(
                dict(det_head_time=time.time()-start)
            )

        if self.training:
            det_out=self._upsample(det_out,imgs.size())
            #torch.Size([1, 6, 640, 640])
            det_loss=self.det_head.loss(det_out,
                                        gt_texts,#torch.Size([1, 640, 640])
                                        gt_kernels,#torch.Size([1, 1, 640, 640])
                                        training_masks,#torch.Size([1, 640, 640])
                                        gt_instances,#torch.Size([1, 640, 640])
                                        gt_bboxes)#torch.Size([1, 201, 4])
            outputs.update(det_loss)

        else:
            det_out=self._upsample(det_out,imgs.size(),4)
            det_res=self.det_head.get_results(det_out,img_metas,cfg)
            outputs.update(det_res)

        return outputs


















