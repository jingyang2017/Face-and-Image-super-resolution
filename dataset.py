import os,time,sys
from glob import glob, iglob
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL

class faces_super(data.Dataset):
    def __init__(self, datasets,transform):
        assert datasets, print('no datasets specified')
        self.transform = transform
        self.img_list = []
        dataset = datasets
        if dataset == 'widerfacetest':
            img_path = './testset/'
            list_name = (glob(os.path.join(img_path, "*.jpg")))
            list_name.sort()
            for filename in list_name:#jpg
                self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index])
        inp64 = inp16.resize((64, 64), resample=PIL.Image.BICUBIC)
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        return data

def get_loader(dataname,bs =1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = faces_super(dataname, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader






















