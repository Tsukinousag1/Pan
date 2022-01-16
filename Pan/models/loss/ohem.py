import torch


def ohem_single(score,gt_text,training_mask):
    pos_num=int(torch.sum(gt_text>0.5))-int(torch.sum((gt_text>0.5)&(training_mask<=0.5)))

    if pos_num==0:
        select_mask=training_mask
        select_mask=select_mask.view(1,select_mask.shape[0],select_mask.shape[1]).float()

        return select_mask

    neg_num=int(torch.sum(gt_text<=0.5))
    neg_num=int(min(pos_num*3,neg_num))
    #neg_num的数量最多为pos_num的3倍

    if neg_num==0:
        select_mask=training_mask
        select_mask=select_mask.view(1,select_mask.shape[0],select_mask[1]).float()

        return select_mask
    #对于score，我们从gt_text中选择位置在哪
    neg_score=score[gt_text<=0.5]
    neg_score_sorted,_=torch.sort(-neg_score)#-4,-3,-2,-1 #1 [2 3 4] #neg_num=3 最大的三个
    threshold=-neg_score_sorted[neg_num-1]#2

    select_mask=((score>=threshold)|(gt_text>0.5))&(training_mask>0.5)
    select_mask=select_mask.reshape(1,select_mask.shape[0],select_mask.shape[1]).float()

    return select_mask

def ohem_batch(scores,gt_texts,training_masks):
    selected_masks=[]

    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i,:,:],gt_texts[i,:,:],training_masks[i,:,:])
        )
    #第o维拼接
    selected_masks=torch.cat(selected_masks,0).float()
    return selected_masks












