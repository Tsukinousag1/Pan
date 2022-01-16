import json
import argparse
import os
import os.path as osp
import random
import sys
import time

import cv2
import numpy as np
import torch
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from utils import AverageMeter

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)

def to_rgb(img):
    img=img.reshape(img.shape[0],img.shape[1],1)
    img=np.concatenate((img,img,img),axis=2)*255
    return img

def train(train_loader,model,optimizer,epoch,start_iter,cfg):
    model.train()

    #meters
    batch_time=AverageMeter()
    data_time=AverageMeter()

    losses=AverageMeter()
    losses_text=AverageMeter()
    losses_kernels=AverageMeter()
    losses_emb=AverageMeter()
    losses_rec=AverageMeter()

    ious_text=AverageMeter()
    ious_kernel=AverageMeter()
    accs_rec=AverageMeter()

    #是否需要检测
    with_rec=hasattr(cfg.model,'recognition_head')

    #start time
    start=time.time()
    for iter,data in enumerate(train_loader):
        #跳过之前部分
        if iter < start_iter:
            print('Skipping iter : %d' % iter)
            sys.stdout.flush()
            continue

        #time cost of data loader
        data_time.update(time.time()-start)

        #adjust learning rate
        adjust_learning_rate(optimizer,train_loader,epoch,iter,cfg)

        #prepare input
        data.update(dict(
            cfg=cfg
        ))
        ####################data
        #print(data['imgs'].shape)torch.Size([1, 3, 640, 640])
        #print(data['gt_texts'].shape)torch.Size([1, 640, 640])
        #print(data['gt_kernels'].shape)torch.Size([1, 1, 640, 640])
        #print(data['training_masks'].shape)torch.Size([1, 640, 640])
        #print(data['gt_instances'].shape)torch.Size([1, 640, 640])
        #print(data['gt_bboxes'].shape)torch.Size([1, 201, 4])
        ###################
        #forward

        outputs=model(**data)


        #detection loss
        loss_text=torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels=torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())

        if 'loss_emb' in outputs.keys():
            loss_emb=torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item())
            loss=loss_text+loss_kernels+loss_emb
        else:
            loss=loss_text+loss_kernels

        iou_text=torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())

        iou_kernel=torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        #recognition loss 是否需要识别
        if with_rec:
            loss_rec=outputs['loss_rec']
            valid=loss_rec>0.5
            if torch.sum(valid)>0:
                loss_rec=torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item())

                loss=loss+loss_rec

                acc_rec=outputs['acc_rec']
                acc_rec=torch.mean(acc_rec[valid])
                accs_rec.update(
                    acc_rec.item(),
                    torch.sum(valid).item())

        losses.update(loss.item())

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-start)

        #update start time
        start=time.time()

        #print log
        if iter % 20==0:
            length=len(train_loader)
            log=f'({iter+1}/{length})' \
                f'LR:{optimizer.param_groups[0]["lr"]:.6f} |' \
                f'Batch:{batch_time.avg:.3f}s | '\
                f'Total:{batch_time.avg*iter/60.0:.0f}min | ' \
                f'ETA:{batch_time.avg*(length-iter)/60.0:.0f}min | ' \
                f'Loss:{losses.avg:.3f} | '\
                f'Loss(text/kernel/emb{"/rec" if with_rec else ""} | ' \
                f'{losses_text.avg:.3f}/{losses_kernels.avg:.3f}/{losses_emb.avg:.3f}' \
                f'{"/"+format(losses_rec.avg,".3f") if with_rec else ""} | '\
                f'IoU(text/kernel):{ious_text.avg:.3f}/{ious_kernel.avg:.3f}'\
                f'{"|ACC rec:"+format(accs_rec.avg,".3f") if with_rec else ""}'

            print(log)
            sys.stdout.flush()



def adjust_learning_rate(optimizer,dataloader,epoch,iter,cfg):
    schedule=cfg.train_cfg.schedule

    if isinstance(schedule,str):
        assert schedule=='polylr','Error:schedule should be polylr!'
        cur_iter=epoch*len(dataloader)+iter
        max_iter_num=cfg.train_cfg.epoch*len(dataloader)
        ##########更新学习率公式
        lr=cfg.train_cfg.lr*(1-float(cur_iter)/max_iter_num)**0.9

    elif isinstance(schedule,tuple):
        lr=cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch <schedule[i]:
                break
            lr=lr*0.1
    #更新optimizer中的学习率参数,得到学习率optimizer.param_groups[0]["lr"]
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
        '''
        optimizer.param_groups： 是长度为2的list，其中的元素是2个字典；
        optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]
        optimizer.param_groups[1]： 好像是表示优化器的状态的一个字典；
        '''


def save_checkpoint(state,checkpoint_path,cfg):
    file_path=osp.join(checkpoint_path,'checkpoint.pth.tar')
    torch.save(state,file_path)

    #训练epoch>100轮次，且epoch%10=0,额外记录保存
    if cfg.data.train.type in ['synth'] or (state['iter']==0 and state['epoch']>cfg.train_cfg.epoch-100 and state['epoch']%10==0):
        file_name='checkpoint_%dep.pth.tar' % state['epoch']
        file_path=osp.join(checkpoint_path,file_name)
        torch.save(state,file_path)


def main(args):
    ###########################################导入config
    #打开config目录文件
    cfg=Config.fromfile(args.config)
    # 读取参数，json.dumps()使字典类型漂亮的输出，indent参数决定添加几个空格
    print(json.dumps(cfg._cfg_dict,indent=4))
    ###########################################

    if args.checkpoint is not None:
        checkpoint_path=args.checkpoint
    else:
        cfg_name,_=osp.splitext(osp.basename(args.config))
        #osp.splitext 'pan_r18_ctw.py' ----> 'pan_r18_ctw' '.py'
        #osp.basename  path='./config/pan_r18_ctw.py'        ----> os.path.basename(path) = pan_r18_ctw.py
        checkpoint_path=osp.join('checkpoints',cfg_name)
        #checkpoints/pan_r18_ctw

    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('Checkpoint path : %s.' % checkpoint_path)
    #Checkpoint path : checkpoints\pan_r18_ctw.
    sys.stdout.flush()

    data_loader=build_data_loader(cfg.data.train)
    train_loader=torch.utils.data.DataLoader(data_loader,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=True,
                                             pin_memory=True)
    #######################################################
    '''check for the data loader
    for batch_idx,imgs in enumerate(train_loader):
        #print(imgs['gt_kernels'].shape)torch.Size([1, 1, 640, 640])
        #print(imgs['gt_texts'].shape)#torch.Size([1, 640, 640])
        #print(imgs['gt_instances'].shape)torch.Size([1, 640, 640])
        #gt_kernels=to_rgb(imgs['gt_kernels'][0][0].numpy().astype(np.uint8))
        #gt_texts=to_rgb(imgs['gt_texts'][0].numpy().astype(np.uint8))
        #gt_instances=to_rgb(imgs['gt_instances'][0].numpy().astype(np.uint8))
        #cv2.imshow('gt_kernels',gt_kernels)
        #cv2.imshow('gt_texts', gt_texts)
        #cv2.imshow('gt_instances',gt_instances)
        #cv2.waitKey(0)
        #cv2.destroyWindow()
        break'''
    #############################################################
    #model
    if hasattr(cfg.model,'recognition_head'):#如果存在属性
        cfg.model.recognition_head.update(dict(
            voc=data_loader.voc,
            char2id=data_loader.char2id,
            id2char=data_loader.id2char,
        ))

    model=build_model(cfg.model)

    if torch.cuda.is_available():
        model=model.cuda()

    #check if model has custom optimizer / loss
    if hasattr(model.modules,'optimizer'):
        optimizer=model.modules.optimizer
    else:
        if cfg.train_cfg.optimizer=='SGD':
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=cfg.train_cfg.lr,
                momentum=0.99,
                weight_decay=5e-4)
        elif cfg.train_cfg.optimizer=='Adam':
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=cfg.train_cfg.lr
            )
    ###########
    start_epoch=0
    start_iter=0
    ###########是否加载预训练模型
    if hasattr(cfg.train_cfg,'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain),'Error: no pretrained weights found!'
        print('Finetuning from pretrained models %s.' % cfg.train_cfg.pretrain)
        checkpoint=torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    ############checkpoint
    if args.resume:
        assert osp.isfile(args.resume),'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint=torch.load(args.resume)
        start_epoch=checkpoint['epoch']
        start_iter=checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch,cfg.train_cfg.epoch):
        print('\nEpoch:[%d | %d]' % (epoch+1,cfg.train_cfg.epoch))

        train(train_loader,model,optimizer,epoch,start_iter,cfg)


        state=dict(
            epoch=epoch+1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict())

        if epoch % 20==0:
            save_checkpoint(state,checkpoint_path,cfg)



if __name__ == '__main__':

    ###############################parser模块
    parser=argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config',default='./config/pan_r18_ctw.py',type=str,help='config file path')
    parser.add_argument('--checkpoint',default='/home/std2021/hejiabang/OCR/PAN/checkpoints',type=str,help='checkpoint path')
    parser.add_argument('--resume',default='/home/std2021/hejiabang/OCR/PAN/checkpoints/pan_r18_ctw/checkpoint.pth.tar',type=str,help='resume')
    args=parser.parse_args()
    ###############################

    main(args)





