# -*- coding: utf-8 -*
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
from  torchsummary import summary

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')


# 构建模型
def get_net():
    num_classes = 2 + 1  # (2个类别) + background
    # num_classes = 2 # (2个类别)
    # 加载经过预训练的模型
    # my_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # my_model = torchvision.models.resnet50(pretrained=True)

    my_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    pre = torch.load(weights_path)
    my_model.load_state_dict(pre)

    # 获取分类器的输入参数的数量
    in_features = my_model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头部替换预先训练好的头部
    my_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return my_model


if __name__=='__main__':
    my_model = get_net()
    device = torch.device('cuda')

    my_model.to(device)
    # print(my_model)
    summary(my_model)