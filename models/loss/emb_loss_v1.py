import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbLoss_v1(nn.Module):
    def __init__(self,feature_dim=4,loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim=feature_dim
        self.loss_weight=loss_weight
        self.delta_v=0.5
        self.delta_d=1.5

        self.weights=(1.0,1.0)

    def forward_single(self,emb,instance,kernel,training_mask,bboxes):
        '''
        :param emb: torch.Size([1, 4, 640, 640])
        emb就是similar vector，shape为(4,w,h)，wh为网络输入的宽和高
        :param instance: torch.Size([1, 640, 640])  #文本实例 instance为i，背景为0
        :param kernel: torch.Size([1, 640, 640])    #shrink后的gt
        :param training_mask: torch.Size([1, 640, 640])
        :param bboxes: torch.Size([1, 201, 4])
        :return:
        '''
        training_mask=(training_mask>0.5).long()
        kernel=(kernel>0.5).long()

        instance=instance*training_mask

        instance_kernel=(instance*kernel).view(-1)
        #torch.Size([409600])
        instance=instance.view(-1)
        #torch.Size([409600])
        emb=emb.view(self.feature_dim,-1)#[4,.]
        #torch.Size([4, 409600])

        #就是挑出tensor中的独立不重复元素[0]
        unique_labels,unique_ids=torch.unique(instance_kernel,sorted=True,return_inverse=True)
        #假设图中有5个文本实例，unique_label==tensor([0, 1, 2, 3, 4，5], device='cuda:0')，0是背景


        num_instance=unique_labels.size(0)#1

        if num_instance<=1:
            return 0
        #假设num_instance=5
        #torch.size([4, 5])
        emb_mean=emb.new_zeros((self.feature_dim,num_instance),dtype=torch.float32)

        for i ,lb in enumerate(unique_labels):
            if lb==0:#背景
                continue
            ind_k=instance_kernel==lb#实例嵌入emb后的核
            emb_mean[:,i]=torch.mean(emb[:,ind_k],dim=1)#计算emb平均值，也即是核的相似度向量

        #torch.size([5])
        l_agg=emb.new_zeros(num_instance,dtype=torch.float32)

        for i ,lb in enumerate(unique_labels):
            if lb==0:
                continue
            ind=instance==lb
            emb_=emb[:,ind]#相似度向量

            dist=(emb_-emb_mean[:,i:i+1]).norm(p=2,dim=0)
            #emb_mean[:, i].shape==torch.Size([4])
            #emb_mean[:, i:i+1].shape==torch.Size([4,1])
            #像素的相似度向量-核的相似度向量做差，然后计算0维度上的二范数
            dist=F.relu(dist-self.delta_v)**2

            l_agg[i]=torch.mean(torch.log(dist+1.0))
        #背景0不计算进去
        l_agg=torch.mean(l_agg[1:])

        #num_instance 等于2就执行以上部分，大于2的时候就执行以下部分
        if num_instance >2:
            #12345 12345 12345 12345 12345
            #11111 22222 33333 44444 55555
            #因为emb_mean为核的相似度
            #torch.size([5, 4])->torch.size([25, 4])每个instance和各个instance一次计算5*5
            emb_interleave=emb_mean.permute(1,0).repeat(num_instance,1)#交错

            #torch.size([5, 4])->torch.size([5, 20])->torch.size([25, 4])每个特征和各个instance 4*5
            emb_band=emb_mean.permute(1,0).repeat(1,num_instance).view(-1,self.feature_dim)

            #对角线都为1，其余都为0 -> 对角线都为0，其余都为1 [5,5]->[25,1]->[25,4]
            #对角线为0，表示自己到自己的dis是0
            mask=(1-torch.eye(num_instance,dtype=torch.int8)).view(-1,1).repeat(1,self.feature_dim)

            mask=mask.view(num_instance,num_instance,-1)#[5,5,4]
            #实例0是背景不用计算
            mask[0,:,:]=0
            mask[:,0,:]=0
            mask=mask.view(num_instance*num_instance,-1)#[25,4]

            dist=emb_interleave-emb_band
            dist=dist[mask>0].view(-1,self.feature_dim).norm(p=2,dim=1)
            dist=F.relu(2*self.delta_d-dist)**2 #dis_min: 1.5*2=3
            l_dis=torch.mean(torch.log(dist+1.0))
        else:#只有一个instance
            l_dis=0

        l_agg=self.weights[0]*l_agg
        l_dis=self.weights[1]*l_dis
        l_reg=torch.mean(torch.log(torch.norm(emb_mean,2,0)+1.0))*0.001
        #l_reg是用来限制emb的模长不能太大，有没有这一项估计差别不是很大。”
        loss=l_agg+l_dis+l_reg
        return loss

    def forward(self,
                emb,
                instance,
                kernel,
                training_mask,
                bboxes,
                reduce=True):
        loss_batch=emb.new_zeros((emb.size(0)),dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i]=self.forward_single(emb[i],instance[i],kernel[i],training_mask[i],bboxes[i])

        loss_batch=self.loss_weight*loss_batch

        if reduce:
            loss_batch=torch.mean(loss_batch)

        return loss_batch



'''
a=[1,2,3,4,5]
print(a[3:4]) [4]
print(a[3]) 4'''















