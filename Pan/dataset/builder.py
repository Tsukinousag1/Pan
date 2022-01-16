import dataset

#{'type': 'PAN_CTW', 'split': 'train', 'is_transform': True, 'img_size': 640, 'short_size': 640, 'kernel_scale': 0.7, 'read_type': 'cv2'}

def build_data_loader(cfg):
    param=dict()
    for key in cfg:
        if key=='type':
            continue
        param[key]=cfg[key]
    #print(param)
    #{'split': 'train', 'is_transform': True, 'img_size': 640, 'short_size': 640, 'kernel_scale': 0.7, 'read_type': 'cv2'}
    #print(cfg.type)
    #PAN_CTW 去到init.py
    #print(*param)
    #split is_transform img_size short_size kernel_scale read_type
    data_loader=dataset.__dict__[cfg.type](**param)
    #<dataset.pan.pan_ctw.PAN_CTW object at 0x000001BA41B834C0>
    return data_loader
