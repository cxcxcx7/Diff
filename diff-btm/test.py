# import torch
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
# from PIL import Image
# import torchvision.transforms as transforms
# from timm.models import create_model
#
# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
#
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.safety_checker = None
# # scheduler = DDPMScheduler()
# # pipe.scheduler = scheduler
# # print(scheduler.timesteps)
# pipe = pipe.to(device)
#
# prompt = "a photo of cabbage butterfly"
# ori_img = Image.open('clean_example/ILSVRC2012_val_00000031.png')
# # ori_img = Image.open('D:\\pycharm\\PyCharm 2023.1.2\\pythonProject\\test2\\111.png')
# transform1 = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# transform2 = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((224, 224)),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# pil_to_tensor = transforms.PILToTensor()
# tensor_to_pil = transforms.ToPILImage()
# # tensor_img = pil_to_tensor(ori_img)
# # print(tensor_img.type)
# tensor_img = transform1(ori_img)
#
#
# pil_img = tensor_to_pil(tensor_img)
# # guidance_scale=0.1
# image = pipe(prompt, init_image=pil_img, strength=0.7).images[0]
#
#
# tensor_image = transform2(image)
# model = create_model("inception_v3", pretrained=True)
# tensor_image = tensor_image.unsqueeze(0)
# print(tensor_image.shape)
# pre = model(tensor_image)
# pre_label = torch.argmax(pre, dim=1)
# print(pre_label)
#
# image.save("114.png")


# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
#
# # 假设clean_features和adversarial_features分别是干净样本和对抗样本的特征表示，维度为(n_samples, n_features)
# # 将干净样本和对抗样本的特征拼接在一起
# all_features = np.vstack((clean_features, adversarial_features))
# # 创建对应的标签，0表示干净样本，1表示对抗样本
# labels = np.hstack((np.zeros(clean_features.shape[0]), np.ones(adversarial_features.shape[0])))
#
# # 使用t-SNE进行降维，降到2维
# tsne = TSNE(n_components=2, random_state=0)
# # 获取降维后的结果
# embedded_features = tsne.fit_transform(all_features)
#
# # 根据标签将降维后的结果分成干净样本和对抗样本两部分
# clean_embedded = embedded_features[labels == 0]
# adversarial_embedded = embedded_features[labels == 1]
#
# # 绘制降维后的结果
# plt.figure(figsize=(10, 6))
# plt.scatter(clean_embedded[:, 0], clean_embedded[:, 1], label='Clean samples')
# plt.scatter(adversarial_embedded[:, 0], adversarial_embedded[:, 1], label='Adversarial samples')
# plt.legend()
# plt.show()

# pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# sns.set()
#
#
# def get_data():
#     """生成聚类数据"""
#     from sklearn.datasets import make_blobs
#     x_value, y_value = make_blobs(n_samples=1000, n_features=40, centers=3, )
#     return x_value, y_value
#
#
# def plot_xy(x_values, label, title):
#     """绘图"""
#     df = pd.DataFrame(x_values, columns=['x', 'y'])
#     df['label'] = label
#     sns.scatterplot(x="x", y="y", hue="label", data=df)
#     plt.title(title)
#     plt.show()
#
#
# def main():
#     x_value, y_value = get_data()
#     # PCA 降维
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     x_pca = pca.fit_transform(x_value)
#     plot_xy(x_pca, y_value, "PCA")
#     # t-sne 降维
#     from sklearn.manifold import TSNE
#     tsne = TSNE(n_components=2)
#     x_tsne = tsne.fit_transform(x_value)
#     plot_xy(x_tsne, y_value, "t-sne")
#
#
# if __name__ == '__main__':
#     main()
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 假设我们已经有一个 PIL 图像
image = Image.open("tmp_img/n01532829_82.JPEG")

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 改变大小为 224x224
    transforms.ToTensor(),          # 转换为 tensor 并且将像素值归一化到 [0, 1]
])

# 应用转换
image_resized = transform(image)

image_resized = image_resized.unsqueeze(0)
save_image(image_resized, 'n01532829_82.png')
print(image_resized.size())


