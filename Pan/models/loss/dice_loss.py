import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,input,target,mask,reduce=True):
        #texts,gt_texts,selected_masks,reduce=False
        batch_size=input.size(0)
        input=torch.sigmoid(input)

        input=input.contiguous().view(batch_size,-1)
        target=target.contiguous().view(batch_size,-1).float()
        mask=mask.contiguous().view(batch_size,-1).float()
        #torch.Size([1, 409600])

        input=input*mask#torch.Size([1, 409600])
        target=target*mask

        a=torch.sum(input*target,dim=1)
        b=torch.sum(input*input,dim=1)+0.001
        c=torch.sum(target*target,dim=1)+0.001
        d=(2*a)/(b+c)
        loss=1-d

        loss=self.loss_weight*loss

        if reduce:
            loss=torch.mean(loss)

        return loss
