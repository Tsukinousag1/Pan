import file_util
import Polygon   as plg
import numpy as np

project_root='C:/Users/86159/PAN/'
data_root='C:/Users/86159/'

pred_root=project_root+'outputs/submit_ctw'
gt_root=data_root+'data/ctw1500/test/text_label_circum/'

def get_pred(path):
    lines=file_util.read_file(path).split('\n')
    bboxes=[]
    for line in lines:
        if line=='':
            continue
        bbox=line.split(',')
        if len(bbox)%2==1:
            print(path)
        bboxes.append(bbox)
    return bboxes

def get_gt(path):
    lines=file_util.read_file(path).split('\n')
    bboxes=[]
    for line in lines:
        if line=='':
            continue
        gt=line.split(',')

        x1=np.int(gt[0])
        y1=np.int(gt[1])

        bbox=[np.int(gt[i]) for i in range(4,32)]
        bbox=np.array(bbox)+([x1,y1]*14)

        bboxes.append(bbox)
    return bboxes

def get_union(pD,pG):
    areaA=pD.area()
    areaB=pG.area()
    return areaA+areaB-get_intersection(pD,pG)

def get_intersection(pD,pG):
    pInt=pD&pG
    if len(pInt)==0:
        return 0
    return pInt.area()

if __name__=='__main__':
    th=0.5

    pred_list=file_util.read_dir(pred_root)

    tp,fp,npos=0,0,0

    for pred_path in pred_list:
        #'/home/std2021/hejiabang/OCR/PAN/outputs/submit_ctw/1001.txt'
        preds=get_pred(pred_path)
        gt_path=gt_root+pred_path.split('/')[-1]
        #C:/Users/86159/data/ctw1500/test/text_label_circum/1496.txt
        gts=get_gt(gt_path)
        #正样本数量，也就是bbox的数量，npos
        npos+=len(gts)

        cover=set()#去重
        for pred_id,pred in enumerate(preds):
            pred=np.array(pred)
            pred=pred.reshape(int(pred.shape[0]/2),2)[:,::-1]
            #<class 'numpy.str_'>
            pred_p=plg.Polygon(tuple(pred))#此处比须传入(N,2)的tuple类型作为参数

            flag=False
            for gt_id,gt in enumerate(gts):
                gt=np.array(gt)
                gt=gt.reshape(int(gt.shape[0]/2),2)
                gt_p=plg.Polygon(tuple(gt))#都转为plg.Polygon形式

                union=get_union(pred_p,gt_p)
                inter=get_intersection(pred_p,gt_p)

                if inter*1.0/union>=th:#IOU>=0.5
                    #匹配成功
                    if gt_id not in cover:
                        flag=True#该pred已经匹配成功
                        cover.add(gt_id)#加入到cover中
            if flag:
                tp+=1.0#truth positive+1，真正例
            else:
                fp+=1.0#如果是False，fasle positive说明匹配失败，预测是假的+1，同理如果遇到了两个预测同一个gt，一样判断为fp

    #print tp fp npos
    precision=tp/(tp+fp)
    #预测为一个一个图继续
    recall=tp/npos
    hmean=0 if (precision+recall)==0 else 2.0*precision*recall/(precision+recall)

    print('p: %.4f , r: %.4f , f: %.4f'% (precision,recall,hmean))
    #p: 0.8494 , r: 0.8018 , f: 0.8249
