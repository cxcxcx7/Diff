import torch
import numpy as np
import torch.nn.functional as F
# import torch_dct as dct
import scipy.stats as st
import torch_dct as dct
from torchvision.utils import save_image
import torch.nn as nn
from PIL import Image
import os
from torchvision import transforms


class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t

    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t

        x = x * 2 - 1

        t = torch.full((x.shape[0],), t).long().to(x.device)

        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

        # print(x_t.min(), x_t.max())

        # si(x_t, 'vis/noised_x.png', to_01=True)

        indices = list(range(t + 1))[::-1]


        for i in indices:
            # out = self.diffusion.ddim_sample(self.model, sample, t)
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0],), i).long().to(x.device))
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def forward(self, x):

        out = self.sdedit(x, self.t)  # [0, 1]
        out = self.classifier(out)
        return out

class SIT:
    def __init__(self, eps, alpha, n_iter, mu):
        self.eps = eps
        self.alpha = alpha
        self.n_iter = n_iter
        self.mu = mu
        self.op = [self.resize, self.scale, self.add_noise,self.dct,self.drop_out]

    # def vertical_shift(self, x):
    #     _, _, w, _ = x.shape
    #     step = np.random.randint(low=0, high=w, dtype=np.int32)
    #     x = torch.roll(x, step, dims=2)
    #     return x
    #
    # def horizontal_shift(self, x):
    #     _, _, _, h = x.shape
    #     step = np.random.randint(low=0, high=h, dtype=np.int32)
    #     x = torch.roll(x, step, dims=2)
    #     return x
    #
    # def vertical_flip(self, x):
    #     return torch.flip(x, dims=(2,))
    #
    # def horizontal_flip(self, x):
    #     return torch.flip(x, dims=(3,))
    #
    # def rotate180(self, x):
    #     return torch.rot90(x, k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def resize(self, x):
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x

    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x)  # torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx  # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255, 16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).cuda()

    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        # y_axis = [0, ] + np.random.choice(list(range(1, h)), 2, replace=False).tolist() + [h, ]
        # x_axis = [0, ] + np.random.choice(list(range(1, w)), 2, replace=False).tolist() + [w, ]

        # y_axis.sort()
        # x_axis.sort()
        x_axis = [0, 74, 149, 224]
        y_axis = [0, 74, 149, 224]

        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([self.blocktransform(x) for _ in range(5)])

    @torch.no_grad()
    def attack(self, model, ori_img, label, diffusion, ddim_model, t):

        net = Denoised_Classifier(diffusion, ddim_model, model, t)

        g_t = torch.zeros_like(ori_img).cuda()

        adv = ori_img.clone()
        for i in range(self.n_iter):
            ddim_img = net.sdedit(adv, t).detach()
            # save_image(ddim_img, "result/y.png")
            tensor_img = 0.7 * adv + 0.3 * ddim_img
            # save_image(tensor_img, "result/fusion.png")
            tensor_img = torch.clamp(tensor_img, 0, 1)
            tensor_img = self.transform(tensor_img)
            # for j in range(5):
            #     save_image(tensor_img[j].unsqueeze(0), 'result/{}.png'.format(j))
            tensor_img.requires_grad = True
            with torch.enable_grad():
                pre = model(tensor_img)
                loss = F.cross_entropy(pre, label.repeat(5).long())
                grads = torch.autograd.grad(loss, tensor_img)[0]
            grad = torch.mean(grads, dim=0, keepdim=True)
            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim=[1, 2, 3], keepdim=True)
            adv = adv.data + self.alpha * torch.sign(g_t)
            delta = torch.clamp(adv - ori_img, -self.eps, self.eps)
            adv = torch.clamp(ori_img + delta, 0, 1).detach()
        return adv

