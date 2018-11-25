from PIL import Image
import PIL
import os
import os.path
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import time

# input_image_size = 224
webface_train_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5409566, 0.41063643, 0.3478864), (0.27481332, 0.23811759, 0.22841948))
])


def default_loader(path, smaller_side_size=256):
    r_img = Image.open(path).convert('RGB')
    size = r_img.size
    h = size[0]
    w = size[1]
    if h > w:
        scale = smaller_side_size/w
    else:
        scale = smaller_side_size/h

    new_h = round(scale*h)
    new_w = round(scale*w)
    r_img = r_img.resize((new_h, new_w), PIL.Image.BILINEAR)
    return r_img


class ImageFilelist(data.Dataset):

    def __init__(self, fname, root_dir=None, transform=None, loader=default_loader, five_crop=False):
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.five_crop = five_crop
        fid = open(fname, 'r')
        lines = fid.readlines()
        imgs = []
        classes = set()
        for line in lines:
            file, label = line.strip().split(' ')
            label = int(label)
            item = (file, label)
            imgs.append(item)
            classes.add(label)

        self.imgs = imgs
        self.classes = list(classes)

    def __getitem__(self, index):
        file, label = self.imgs[index]

        if self.root_dir is not None:
            path = os.path.join(self.root_dir, file)
        else:
            path = file

        if not self.five_crop:
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            label = int(label)

            return img, label, path
        else:
            imgs = self.loader(path)
            lst_img = []
            for img in imgs:
                if self.transform is not None:
                    img = self.transform(img)
                    lst_img.append(img)
            label = int(label)
            return lst_img, label, path

    def __len__(self):
        return len(self.imgs)
