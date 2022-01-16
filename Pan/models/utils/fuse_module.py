'''
原因：当前CNN卷积层的基本组成单元为：Conv+BN+ReLu三剑客，这几乎成为标配。但其实在网络的推理阶段，
可以将BN层的运算融合到Conv层中，减少运算量，加速推理。
本质上是修改了卷积核的参数，在不增加Conv层计算量的同时，略去了BN层的计算量。
'''
import torch
import torch.nn as nn

def fuse_con_bn(conv,bn):
    conv_w=conv.weight
    conv_b=conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

    factor=bn.weight/torch.sqrt(bn.running_var+bn.eps)

    conv.weight=nn.Parameter(conv_w*factor.reshape([conv.out_channels,1,1,1]))

    conv.bias=nn.Parameter((conv_b-bn.running_mean)*factor+bn.bias)
    return conv

def fuse_module(m):
    last_conv=None
    last_conv_name=None

    for name,child in m.named_children():
        if isinstance(child,(nn.BatchNorm2d,nn.SyncBatchNorm)):
            if last_conv is None:#only fuse BN that is after Conv
                continue
            fused_conv=fuse_con_bn(last_conv,child)
            m._modules[last_conv_name]=fused_conv

            m._modules[name]=nn.Identity()
            last_conv=None
        elif isinstance(child,nn.Conv2d):
            last_conv=child
            last_conv_name=name

        else:
            fuse_module(child)

    return m












