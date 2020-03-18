import json
import random
from flyai.dataset import Dataset
from model import Model,device,MODEL_PATH,TORCH_MODEL_NAME
from flyai.processor.download import check_download
from path import DATA_PATH
import numpy as np
import os
import requests
from net import get_net
import torchvision
import torch

# 根据 recall 和 precision 值来计算 ap
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(groundtruth, prediction, classname, ovthresh=0.5, use_07_metric=False):
    '''
    :param groundtruth: 格式为：[{'boxes':[], 'labels':[], 'image_id':[]}, ...]
    :param prediction: 格式为：[[<image id> <confidence> <left> <top> <right> <bottom>], ...]
    :param classname: 计算的是当前类别的 ap
    :param ovthresh: iou的阈值
    :param use_07_metric: 标准选择，是否是之前的标准
    :return:
    '''
    imagenames = [i['image_id'][0] for i in groundtruth]
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        for i in groundtruth:
            if i['image_id'][0] == imagename:
                bbox = []
                for j in range(len(i['labels'])):
                    if i['labels'][j] == classname:
                        bbox.append(i['boxes'][j])
                bbox = np.array(bbox)
                difficult = np.zeros((len(bbox))).astype(np.bool) # 这里默认所有目标框全部为易辨别物体，用于参与后续计算
                det = [False] * len(bbox)
                npos = npos + sum(~difficult)
                class_recs[imagename] = {'bbox': bbox,
                                         'difficult': difficult,
                                         'det': det}
                break

    # load prection
    image_ids = [x[0] for x in prediction]
    confidence = np.array([float(x[1]) for x in prediction])
    BB = np.array([[float(z) for z in x[2:]] for x in prediction])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def get_json(url, is_log=False):
    try:
        response = requests.get(url=url)
        if is_log:
            print("server code ", response, url)
        return response.json()
    except:
        return None

# '''
# 不需要修改
# '''
# if "https" in sys.argv[1]:
#     data_id = sys.argv[1].split('/')[4]
# else:
#     data_id = sys.argv[1]
# data_path = get_json("https://www.flyai.com/get_evaluate_command?data_id=" + data_id)
# check_download(data_path['command'], DATA_PATH, is_print=False)

def eval_one_batch(x_test,y_test, model ):
    # randnum = random.randint(0, 100)
    # random.seed(randnum)
    # random.shuffle(x_test)
    # random.seed(randnum)
    # random.shuffle(y_test)

    # 通过模型得到预测的结果，格式为：[[<image id> <confidence> <left> <top> <right> <bottom>], ...]
    preds = model.predict_all(x_test)

    # 加载标签 [{'boxes':[], 'labels':[], 'image_id':[]}, ...]
    targets = []
    for i in range(len(y_test)):
        label_path = y_test[i]['label_path']  # label/019646.jpg.txt
        boxes = []
        labels = []
        image_id = []
        image_id.append(x_test[i]['image_path'])
        with open(os.path.join(DATA_PATH, label_path)) as f:
            for line in f.readlines():
                # 1954.7443195924375,695.1497671989313,1984.659514688955,738.4779589540301,1933
                temp = line.strip().split(',')
                xmin = int(float(temp[0]))
                ymin = int(float(temp[1]))
                xmax = int(float(temp[2]))
                ymax = int(float(temp[3]))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(temp[4]))
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        targets.append(target)

    '''
    if不需要修改
    '''
    if len(y_test) != len(x_test):
        result = dict()
        result['score'] = 0
        result['label'] = "评估违规"
        result['info'] = ""
        print(json.dumps(result))
    else:
        '''
        在下面实现不同的评估算法
        '''
        # 开始计算最后得分
        sum_ap = 0
        all_labels = [i for i in range(2)]  # 所有目标类别

        # 以下是我自己加的

        for label in all_labels:  # 逐个类别计算ap
            prediction1 = []  # 在计算 ap 的时候，需要把prediction按照最后预测的类别进行筛选
            for pred in preds:
                if pred[3] == label:
                    prediction1.append([pred[0], pred[1], pred[2][0], pred[2][1], pred[2][2], pred[2][3]])
            if len(prediction1) != 0:  # 当包含预测框的时候，进行计算ap值
                rec, prec, ap = voc_eval(targets, prediction1, label)
            else:
                ap = 0
            sum_ap += ap
        map = sum_ap / len(all_labels)

        result = dict()
        result['score'] = round(map * 100, 2)
        result['label'] = "The Score is MAP."
        result['info'] = ""
        print(json.dumps(result))

    return result

if __name__=="__main__":

    dataset = Dataset()
    model = Model(dataset)
    try:
        x_test, y_test = dataset.evaluate_data_no_processor("test.csv")
        print('eval.py use test.csv')
    except:
        x_test, y_test = dataset.evaluate_data_no_processor("dev.csv")
        print('eval.py use dev.csv')
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(x_test)
    random.seed(randnum)
    random.shuffle(y_test)

    # 通过模型得到预测的结果，格式为：[[<image id> <confidence> <left> <top> <right> <bottom>], ...]
    preds = model.predict_all(x_test)

    # 加载标签 [{'boxes':[], 'labels':[], 'image_id':[]}, ...]
    targets = []
    for i in range(len(y_test)):
        label_path = y_test[i]['label_path'] # label/019646.jpg.txt
        boxes = []
        labels = []
        image_id = []
        image_id.append(x_test[i]['image_path'])
        with open(os.path.join(DATA_PATH, label_path)) as f:
            for line in f.readlines():
                # 1954.7443195924375,695.1497671989313,1984.659514688955,738.4779589540301,1933
                temp = line.strip().split(',')
                xmin = int(float(temp[0]))
                ymin = int(float(temp[1]))
                xmax = int(float(temp[2]))
                ymax = int(float(temp[3]))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(temp[4]))
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        targets.append(target)

    '''
    if不需要修改
    '''
    if len(y_test) != len(x_test):
        result = dict()
        result['score'] = 0
        result['label'] = "评估违规"
        result['info'] = ""
        print(json.dumps(result))
    else:
        '''
        在下面实现不同的评估算法
        '''
        # 开始计算最后得分
        sum_ap = 0
        all_labels = [i for i in range(2)] # 所有目标类别

        # 以下是我自己加的


        for label in all_labels: # 逐个类别计算ap
            prediction1 = []    # 在计算 ap 的时候，需要把prediction按照最后预测的类别进行筛选
            for pred in preds:
                if pred[3] == label:
                    prediction1.append([pred[0], pred[1], pred[2][0], pred[2][1], pred[2][2], pred[2][3]])
            if len(prediction1) != 0: # 当包含预测框的时候，进行计算ap值
                rec, prec, ap = voc_eval(targets, prediction1, label)
            else:
                ap = 0
            sum_ap += ap
        map = sum_ap / len(all_labels)

        result = dict()
        result['score'] = round(map * 100, 2)
        result['label'] = "The Score is MAP."
        result['info'] = ""
        print(json.dumps(result))