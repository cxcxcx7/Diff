import PIL.Image
import torch
# from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from attack import MIFGSM, SIT
from torch.utils.data import DataLoader
from dataset import get_img
from timm.models import create_model
import torch.nn as nn
from tqdm import tqdm
import os
from load_dm import get_imagenet_dm_conf


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


if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = get_img('tmp_img', transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    attacker = SIT.SIT(eps=16/255, alpha=1.6/255, n_iter=10, mu=1.0)

    ddim_model, diffusion = get_imagenet_dm_conf(device=0, respace='ddim50')
    model = create_model('inception_v3', pretrained=True)
    model.eval()
    model = WrapModel(model).cuda()
    for i, data in enumerate(tqdm(data_loader)):
        print("正在处理第{}张图片".format(i+1))
        img = data[0].cuda()
        label = data[1].cuda()
        img_name = data[3][0]
        pre = model(img)
        x_adv = attacker.attack(model, img, label, diffusion, ddim_model, t=10)
        save_image(x_adv, 'adversarial_example/{}'.format(img_name))


