import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class get_img(Dataset):
    def __init__(self, img_root, transform=None):
        self.transform = transform
        query_paths = []
        root_paths = []
        for root, dirs, files in os.walk(img_root):
            for i in files:
                root_paths.append(os.path.join(root, i))
                query_paths.append(i.split('.')[0]+'.JPEG')
        self.query_paths = query_paths
        self.root_paths = root_paths
        with open('image_name_to_class_id_and_name.json') as ipt:
            self.json_info = json.load(ipt)
    def __len__(self):
        return len(self.root_paths)

    def __getitem__(self, index):
        root_path = self.root_paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        img = Image.open(root_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id, class_name, query_path.split('.')[0]+'.png'


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = get_img('result', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, data in enumerate(dataloader):
        img = data[0].cuda()
        label = data[1]
        save_image(img, 'test/{}'.format(data[3][0]))
