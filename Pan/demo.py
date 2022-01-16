import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
from models import build_model
from mmcv import Config


def get_img(img_path,read_type='cv2'):
    try:
        if read_type=='cv2':
            img=cv2.imread(img_path)
            img=img[:,:,[2,1,0]]
        elif read_type=='pil':
            img=np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def scale_aligned_short(img,short_size=640):
    h,w=img.shape[0:2]
    scale=short_size*1.0/min(h,w)
    h=int(h*scale+0.5)
    w=int(w*scale+0.5)
    if h%32!=0:
        h=h+(32-h%32)
    if w%32!=0:
        w=w+(32-w%32)
    img=cv2.resize(img,dsize=(w,h))
    return img

img_path='/home/std2021/hejiabang/OCR/PAN/1159.jpg'
checkpoint_path='/home/std2021/hejiabang/OCR/PAN/checkpoints/pan_r18_ctw/checkpoint.pth.tar'

img=get_img(img_path,'cv2')

img_meta=dict(
    org_img_size=np.array(img.shape[:2])
)
img_result=img
#调整图像大小
img=scale_aligned_short(img,short_size=640)

img_meta.update(dict(
    img_size=np.array(img.shape[:2])
))

img=Image.fromarray(img)
img=img.convert('RGB')
img=transforms.ToTensor()(img)
img=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])(img)
img=torch.unsqueeze(img,dim=0)

data=dict(
    imgs=img,
    img_metas=img_meta
)
cfg=Config.fromfile('./config/pan_r18_ctw.py')

model=build_model(cfg.model)
model=model.cuda()



if checkpoint_path is not None:
    if os.path.isfile(checkpoint_path):
        print("Loading model and optimizer from checkpoint '{}'".format(
            checkpoint_path))
        sys.stdout.flush()

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))

model.eval()
data['imgs']=data['imgs'].cuda()

cfg.report_speed=None
data.update(dict(
            cfg=cfg
        ))

with torch.no_grad():
    outputs=model(**data)

bboxes=outputs['bboxes']


img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)


for i in range(len(bboxes)):
    contour=bboxes[i].reshape(-1,2)
    cv2.drawContours(img_result,[contour],-1,(200,0,30),2)

cv2.imshow('img_result',img_result)

cv2.waitKey(0)
cv2.destroyWindow()








