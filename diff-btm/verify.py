import os.path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from load_model import load_model
from torchvision import transforms
from dataset import get_img
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from timm.models import create_model
from timm import create_model
import torch.nn as nn


class WrapModel(nn.Module):
    def __init__(self, model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(WrapModel, self).__init__()

        self.model = model
        self.mean = torch.as_tensor(mean).view(1, 3, 1, 1)
        self.std = torch.as_tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        x = (x - self.mean.detach()) / self.std.detach()
        return self.model(x)

# vit_base_patch16_224
# pit_b_224
# cait_s24_224
# visformer_small
# deit_tiny_patch16_224
# tnt_s_patch16_224
# inception_v3
# inception_v4
# inception_resnet_v2
# resnetv2_152x2_bit
# vgg16
# resnet50


def get_ASR(model, model_name, data_loader):
    cnt = 0
    for index, data in enumerate(data_loader):
        img = data[0].cuda()
        label = data[1].cuda()
        pre = model(img)
        pre_label = torch.argmax(F.softmax(pre, dim=1), dim=1)
        # print(label,pre_label)
        # cnt += pre_label.eq(label.view_as(pre_label)).sum().item()
        if pre_label != label:
            cnt += 1
    print('{} ASR:{:.3f}'.format(model_name, cnt / 1000))


if __name__ == '__main__':
    # model = load_model('vgg16')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = get_img('adversarial_example/DIFGSM/ensemble', transform)
    # dataset = get_img('adversarial_example/TIFGSM/ensemble', transform)
    # dataset = get_img('adversarial_example/SINIFGSM/ensemble', transform)
    dataset = get_img('adversarial_example/SIT/ensemble', transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    models = ['inception_v3', 'inception_v4', 'inception_resnet_v2', 'resnet101']
    for models_name in models:
        model = create_model(models_name, pretrained=False)
        model.load_state_dict(torch.load(os.path.join('model_preweights', models_name+'.pth')))
        model.eval()
        model = WrapModel(model).cuda()
        get_ASR(model, models_name, data_loader)
 