# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import datetime
from time import clock
import os
import cv2
import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH, DATA_PATH
import torchvision
from net import get_net
from eval import eval_one_batch, voc_ap, voc_eval
import json

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
下载模版之后需要把当前样例项目的app.yaml复制过去哦～
第一次使用请看项目中的：FLYAI项目详细文档.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=3, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
进入Dataset类中可查看方法说明
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)


# 自定义获取数据的方式
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_path_list, label_path_list):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_PATH, self.img_path_list[idx]['image_path'])
        label_path = os.path.join(DATA_PATH, self.label_path_list[idx]['label_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        with open(label_path) as f:
            for line in f.readlines():
                # 237,53,291,114,0
                temp = line.strip().split(',')
                xmin = int(float(temp[0]))
                ymin = int(float(temp[1]))
                xmax = int(float(temp[2]))
                ymax = int(float(temp[3]))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(temp[4]))  # （标签是从0开始，0代表没有佩戴口罩, 1代表佩戴口罩）由于默认0为背景类，所以剩下类别+1
                areas.append((xmax - xmin) * (ymax - ymin))
                iscrowd.append(0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.tensor([idx])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        return torchvision.transforms.ToTensor()(img), target

    def __len__(self):
        return len(self.img_path_list)


def collate_fn(batch):
    return tuple(zip(*batch))


def val_part(my_model,images_val,targets_val,this_dataset):
    my_model.eval()

    images_val_2 = list(images_val.to(device) for images_val in images_val)
    targets_val_2 = [{k: v.to(device) for k, v in t.items()} for t in targets_val]
    targets_val_3 = [{k: v.tolist() for k, v in t.items()} for t in targets_val]
    # targets_val_3 = [{k: v.tolist() for k, v in t['image_id']} for t in targets_val_3]
    for i in range(len(targets_val_3)):
        # print(train_dataset.img_path_list[targets_val_3[i]['image_id'][0]]['image_path'])
        # print(targets_val_3[i]['image_id'])
        # targets_val_3[i]['image_id'] = train_dataset.img_path_list[targets_val_3[i]['image_id'][0]]['image_path']
        targets_val_3[i].update({'image_id': [this_dataset.img_path_list[targets_val_3[i]['image_id'][0]]['image_path']]})
        # print(targets_val_3[i]['image_id'])

    val_dict = my_model(images_val_2, targets_val_2)

    # 1、把loss_dict 转化为predict_all（）return的结果的格式 √
    # 2、调用eval_one_batch √
    # 3、打印评估结果和save best map

    preds = []
    for i in range(len(val_dict)):
        for j in range(len(val_dict[i]['scores'])):
            preds.append([
                            targets_val_3[i]['image_id'][0],
                # x_train[0]['image_path'],
                          val_dict[i]['scores'][j],
                          val_dict[i]['boxes'][j][0].cpu().detach().item(),
                          val_dict[i]['boxes'][j][1].cpu().detach().item(),
                          val_dict[i]['boxes'][j][2].cpu().detach().item(),
                          val_dict[i]['boxes'][j][3].cpu().detach().item(),
                          ])  # 这里pred['labels'][j]-1 用于与标签对应 0-没有佩戴，1-有佩戴

    # eval_one_batch(eval_labels)

    # 开始计算最后得分
    sum_ap = 0
    all_labels = [i for i in range(2)]  # 所有目标类别

    for label in all_labels:  # 逐个类别计算ap

        if len(val_dict) != 0:  # 当包含预测框的时候，进行计算ap值
            rec, prec, ap = voc_eval(targets_val_3, preds, label)
        else:
            ap = 0
        sum_ap += ap
    map = sum_ap / len(all_labels)

    # result = dict()
    # result['score'] = round(map * 100, 2)
    # result['label'] = "The Score is MAP."
    # result['info'] = ""
    # print(json.dumps(result))
    print('mAP is %.2f' % map)
    return map


# 获取所有原始数据
x_train, y_train, x_val, y_val = dataset.get_all_data()  # 示例： [{'img_path': 'img/019646.jpg'}, ...] [{'label_path': 'label/019646.jpg.txt'}, ...]
# 构建自己的数据加载器
train_dataset = MaskDataset(x_train, y_train)
valid_dataset = MaskDataset(x_val, y_val)

#  批大小
train_batch_size = args.BATCH
valid_batch_size = args.BATCH
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

dataiter = iter(valid_data_loader)

'''
实现自己的网络机构
'''
my_model = get_net()
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
    print('pytorch use cuda')
else:
    device = 'cpu'
    print('pytorch use cpu')

device = torch.device(device)
my_model.to(device)

TORCH_MODEL_NAME = "model.pkl"
lr = 1e-4
params = [p for p in my_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

'''
dataset.get_step() 获取数据的总迭代次数
'''

lowest_loss = 100
lowest_batch_loss = 100
highest_score_map = 0
for epoch in range(args.EPOCHS):
    time_1 = clock()
    batch_step = 0
    epoch_loss = 0
    '''
    1.train
    '''
    for images, targets in train_data_loader:
        for param in my_model.parameters():
            param.requires_grad = True
        my_model.train()
        batch_step += 1
        # print(images) # tensor的图片数字整列
        # print(targets) # MaskDataset里的target ， 有boxes,labels,image_id,area,iscrowd
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = my_model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        temp_batch_loss = losses.cpu().detach().numpy()
        # print('train epoch: %d/%d, batch: %d/%d, batch_loss: %f' % (
        #     epoch + 1, args.EPOCHS, batch_step, len(train_data_loader), temp_batch_loss))
        epoch_loss += temp_batch_loss
        '''
        2. val
        '''
        for param in my_model.parameters():
            param.requires_grad = False
        train_score = val_part(my_model, images, targets,train_dataset)
        try:
            images_val, targets_val = dataiter.next()
        except StopIteration:
            print('valid_data_loader 遍历完毕，从头再来')
            dataiter = iter(valid_data_loader)
            images_val, targets_val = dataiter.next()

        # print('images_val', images_val)
        # print('targets_val', targets_val)
        val_score = val_part(my_model, images_val, targets_val,valid_dataset)

        # 改到val去保存best
        if highest_score_map < val_score:
            highest_score_map = val_score
            print('save best by map: %.4f'%highest_score_map)
            torch.save(my_model.state_dict(), os.path.join(MODEL_PATH, TORCH_MODEL_NAME))

    epoch_loss = epoch_loss / len(train_data_loader)
    print('train epoch: %d, loss: %f,best val mAP score:%.2f' % (epoch + 1, epoch_loss,highest_score_map*100))


    '''
    3. 调整学习率
    '''
    lr_scheduler.step()

    '''
    4. save best
    '''
    # if epoch_loss < lowest_loss:
    #     lowest_loss = epoch_loss
    #     # 保存模型
    #     torch.save(my_model.state_dict(), os.path.join(MODEL_PATH, TORCH_MODEL_NAME))
    #     print("lowest loss: %f" % lowest_loss)

    cost_time = clock() - time_1
    need_time_to_end = datetime.timedelta(
        seconds=(args.EPOCHS - epoch - 1) * int(cost_time))
    print('耗时：%d秒,预估还需' % (cost_time), need_time_to_end)