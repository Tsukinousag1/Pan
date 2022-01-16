import models

def build_backbone(cfg):

    param=dict()
    for key in cfg:
        if key=='type':
            continue
        param[key]=cfg[key]

    #print(cfg.type)
    #print(*param) pretrained
    #{'pretrained': True}
    #print(models.backbone.__dict__[cfg.type]) 此处对应的时dict['resnet18']
    #<function resnet18 at 0x000001C7A55A79D0>
    #print(models.backbone.__dict__)
    # {'type': 'PAN_CTW', 'split': 'train', 'is_transform': True, 'img_size': 640, 'short_size': 640, 'kernel_scale': 0.7, 'read_type': 'cv2'}
    #backbone=models.backbone.__dict__[cfg.type](param.values())
    backbone = models.backbone.__dict__[cfg.type](**param)
    #functional 传入parma的value值

    return backbone