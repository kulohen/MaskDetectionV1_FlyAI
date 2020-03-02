# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from path import MODEL_PATH, DATA_PATH
from net import get_net
import cv2
import torchvision

TORCH_MODEL_NAME = "model.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)

    def predict_all(self, datas):
        my_model = get_net()
        my_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, TORCH_MODEL_NAME)))
        my_model.to(device)
        for param in my_model.parameters():
            param.requires_grad = False
        my_model.eval()
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            img = cv2.imread(os.path.join(DATA_PATH, x_data[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(img)
            pred = my_model([img.to(device)])[0]
            for (k, v) in pred.items():
                pred[k] = v.cpu().numpy()
            for j in range(len(pred['labels'])):
                labels.append([x_data[0], pred['scores'][j], pred['boxes'][j], pred['labels'][j]-1]) # 这里pred['labels'][j]-1 用于与标签对应 0-没有佩戴，1-有佩戴
        ''' 关于返回的说明： 
        返回labels为一个list集合，其中每一项样例为['img/000278.jpg', 0.99910635, [117.190445, 108.4803, 163.91493, 222.29749], 1]，包括：
            x_data[0]：图片的相对路径
            pred['scores'][j]：预测框的置信度
            pred['boxes'][j]：预测框的坐标 [x_min, y_min, x_max, y_max]
            pred['labels'][j]：预测框所属类别
        评估指标为map。
        '''
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))