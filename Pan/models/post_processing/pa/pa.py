import numpy as np
import queue
import cv2


def _pa(kernel, emb, label, cc, label_num, min_area=0):
    '''
    :param kernel: (1, 248, 160)
    :param emb: (4, 248, 160)
    :param label: (248,160)
    :param cc: (248,160) 完整预测图中的label信息
    :param label_num: 5
    :param min_area: 0
    :return:
    '''
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    area = np.full((label_num,), -1, dtype=np.float32)
    flag = np.zeros((label_num,), dtype=np.int32)
    inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    p = np.zeros((label_num, 2), dtype=np.int32)

    max_rate = 1024
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)  # 614.0

        if area[i] < min_area:  # 0
            label[ind] = 0
            continue

        px, py = np.where(ind)
        #对于第i个label，我们以最左上角的点来表示边缘点
        p[i] = (px[0], py[0])  # px[0]==min(px), py[0]==min(py)

        for j in range(1, i):#遍历i之前的label
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:  # 完整的text预测图中没有把两个kernel合并成一个
                continue
            #如果i和j在放大后合并为一个
            rate = area[i] / area[j]
            #如果两个area的比例悬殊，小区域部分
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)

                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    que = queue.Queue(maxsize=0)
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        que.put((x, y, l))
        pred[x, y] = l

    while not que.empty():
        (x, y, l) = que.get()
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                continue
            if kernel[0, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:  # 完整text预测图中这个点值为0或者已经扩充过了
                continue
            if flag[l] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[l]) > 3:  # 论文里是6
                continue

            que.put((tmpx, tmpy, l))
            pred[tmpx, tmpy] = l

    return pred


def pa(kernels, emb, min_area=0):  #(2, 248, 160) (4, 248, 160)
    # kernels[0]是预测的text完整图，kernels[1]是预测的以0.5比例shrink的kernel图
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4)
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4)
    # label_num包含了背景，实际要-1
    return _pa(kernels[:-1], emb, label, cc, label_num, min_area)
    #(1, 248, 160) (4, 248, 160) (248,160) (248,160) 5  0
    # kernels[0].shape=(248, 160), kernels[:-1].shape=(1, 248, 160)