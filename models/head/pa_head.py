import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..loss import build_loss,iou,ohem_batch
from ..post_processing import pa

class PA_Head(nn.Module):
    def __init__(self,in_channels,hidden_dim,num_classes,
                 loss_text,loss_kernel,loss_emb):
        #hidden:128 num_classes:6
        super(PA_Head, self).__init__()
        self.conv1=nn.Conv2d(in_channels,
                             hidden_dim,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.bn1=nn.BatchNorm2d(hidden_dim)
        self.relu1=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(hidden_dim,
                             num_classes,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        self.text_loss=build_loss(loss_text)#Diceloss
        self.kernel_loss=build_loss(loss_kernel)#Diceloss
        self.emb_loss=build_loss(loss_emb)#emb_loss

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,f):
        ##torch.Size([1, 512, 160, 160])
        out=self.conv1(f)
        #torch.Size([1, 128, 160, 160])
        out=self.relu1(self.bn1(out))
        out=self.conv2(out)
        #torch.Size([1, 6, 160, 160])
        return out

    def get_results(self,out,img_meta,cfg):
        outputs=dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start=time.time()

        score=torch.sigmoid(out[:,0,:,:])
        kernels=out[:,:2,:,:]>0

        text_mask=kernels[:,:1,:,:]
        kernels[:,1:,:,:]=kernels[:,1:,:,:]*text_mask

        emb=out[:,2:,:,:]

        emb=emb*text_mask.float()

        score=score.data.cpu().numpy()[0].astype(np.float32)
        kernels=kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb=emb.cpu().numpy()[0].astype(np.float32)

        #pa
        '''print('################################\n')
        print(kernels.shape)
        print('################################\n')
        kernels=kernels[1]
        kernels = kernels.reshape(kernels.shape[0], kernels.shape[1], 1)
        kernels = np.concatenate((kernels, kernels, kernels), axis=2) * 255
        cv2.imshow('kernels',kernels)
        cv2.waitKey(0)
        cv2.destroyWindow()'''

        label= pa(kernels,emb)

        #img_size 不要乱用.astype(np.uint8)
        org_img_size=img_meta['org_img_size']
        img_size=img_meta['img_size']

        label_num=np.max(label)+1

        label=cv2.resize(label,(img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)
        score=cv2.resize(score,(img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_post_time=time.time()-start))

        scale=(float(org_img_size[1])/float(img_size[1]),float(org_img_size[0])/float(img_size[0]))

        bboxes=[]
        scores=[]
        for i in range(1,label_num):
            ind=label==i
            points=np.array(np.where(ind)).transpose((1,0))

            if points.shape[0]<cfg.test_cfg.min_area:
                label[ind]=0
                continue

            score_i=np.mean(score[ind])
            if score_i<cfg.test_cfg.min_score:
                label[ind]=0
                continue

            if cfg.test_cfg.bbox_type=='rect':
                rect=cv2.minAreaRect(points[:,::-1])
                bbox=cv2.boxPoints(rect)*scale
            elif cfg.test_cfg.bbox_type=='poly':
                binary=np.zeros(label.shape,dtype='uint8')
                binary[ind]=1
                contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                bbox=contours[0]*scale

            bbox=bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores))
        return outputs

    def loss(self,out,gt_texts,gt_kernels,training_masks,gt_instances,gt_bboxes):
        '''
        :param out: torch.Size([1, 6, 640, 640])
        :param gt_texts: torch.Size([1, 640, 640])
        :param gt_kernels: torch.Size([1, 1, 640, 640])
        :param training_masks: torch.Size([1, 640, 640])
        :param gt_instances: torch.Size([1, 640, 640])
        :param gt_bboxes: torch.Size([1, 201, 4])
        :return:
        '''
        #output
        texts=out[:,0,:,:]#0是texts/是文本的score  torch.Size([1, 640, 640])
        kernels=out[:,1:2,:,:]#1是kernels
        embs=out[:,2:,:,:]#2，3，4，5是emb

        #text loss
        #在线难例挖掘求文本loss
        selected_masks=ohem_batch(texts,gt_texts,training_masks)
        #torch.Size([1, 640, 640])选择texts中的文本区域
        loss_text=self.text_loss(texts,gt_texts,selected_masks,reduce=False)

        iou_text=iou((texts>0).long(),
                     gt_texts,
                     training_masks,
                     reduce=False)

        losses=dict(
            loss_text=loss_text,
            iou_text=iou_text)

        #kernel loss求kernel loss，用[1]和gt_kernel计算Diceloss
        loss_kernels=[]
        selected_masks=gt_texts*training_masks
        for i in range(kernels.size(1)):
            kernel_i=kernels[:,i,:,:]
            gt_kernel_i=gt_kernels[:,i,:,:]
            loss_kernel_i=self.kernel_loss(
                kernel_i,
                gt_kernel_i,
                selected_masks,
                reduce=False
            )
            loss_kernels.append(loss_kernel_i)

        #print(torch.stack(loss_kernels,dim=1))tensor([[0.5000]], grad_fn=<StackBackward>)
        loss_kernels=torch.mean(torch.stack(loss_kernels,dim=1),dim=1)

        iou_kernel=iou((kernels[:,-1,:,:]>0).long(),
                       gt_kernels[:,-1,:,:],
                       training_masks*gt_texts,
                       reduce=False)

        losses.update(dict(loss_kernels=loss_kernels,
                           iou_kernel=iou_kernel))

        #embedding loss
        loss_emb=self.emb_loss(
            embs,#torch.Size([1, 4, 640, 640])
            gt_instances,# torch.Size([1, 640, 640])
            gt_kernels[:,-1,:,:],#torch.Size([1, 640, 640])
            training_masks,#torch.Size([1, 640, 640])
            gt_bboxes,#torch.Size([1, 201, 4])
            reduce=False
        )
        losses.update(dict(loss_emb=loss_emb))

        return losses












