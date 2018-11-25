import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import pickle
import os
import h5py
import numpy as np
import resnet
import time
import argparse

parser = argparse.ArgumentParser(
                                description='training hyper parameters')
parser.add_argument('--net_type',
                    type=str,
                    default='resnet_34_coso',
                    help='')
parser.add_argument('--gpus',
                    type=str,
                    default='3',
                    help='')
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='batch size')
parser.add_argument('--ckp_path',
                    type=str,
                    default='./data/WebFace/model/resnet_34_coso/step_50000.model',
                    help='')
parsed = parser.parse_args()
net_type = parsed.net_type
ckp_path = parsed.ckp_path
batch_size = parsed.batch_size

os.environ['CUDA_VISIBLE_DEVICES'] = parsed.gpus
data_dir = './data/IJB_A/img_cropped_extend'
save_dir = './data/IJB_A/deep_feat'
subs_path = './data/IJB_A/split_data/lst_template.txt'
lst_subjects = pickle.load(open(subs_path, 'rb'))

hf = h5py.File(os.path.join(save_dir, 'IJB_A_{}.h5'.format(net_type)), 'w')
num_classes = 10575 - 25
print(net_type)
print('resume from {}'.format(ckp_path))


def default_loader(path, no_operator=False, smaller_side_size=256):

    r_img = Image.open(path).convert('RGB')
    if no_operator:
        return r_img
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


class VideoLoader_from_lst(data.Dataset):
    def __init__(self, root_dir, lst_video, transform=None, target_transform=None, loader=default_loader):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.lst_video = lst_video

    def __getitem__(self, index):
        video = self.lst_video[index]
        video_name = video.video_name
        lst_path = video.lst_frame_paths
        imgs = []
        media_labels = np.zeros(shape=(len(lst_path), 1), dtype=np.int)
        idx = 0
        for path in lst_path:
            path = path.replace('\\', '/')
            img_path = os.path.join(self.root_dir, path)
            if not os.path.exists(img_path):
                print(img_path)
                continue
            img = self.loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            if 'img' in path:
                media_labels[idx] = 0
            elif 'frame' in path:
                media_labels[idx] = 1
            else:
                media_labels[idx] = -1
            idx += 1
        return imgs, media_labels, video_name

    def __len__(self):
        return len(self.lst_video)


net = []
input_image_size = 224
train_transforms = transforms.Compose([
    transforms.CenterCrop(input_image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5409566, 0.41063643, 0.3478864), (0.27481332, 0.23811759, 0.22841948))
])
if net_type == 'resnet_34':
    net = resnet.resnet34(num_classes=num_classes)
    checkpoint = torch.load(ckp_path)
    resume_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net_state_dict'])
    net = net.cuda()
    net = net.eval()

if net_type == 'resnet_34_coso':
    conv_net, spd_net = resnet.resnet34_cov(num_classes=num_classes, norm_type='T', num_iter=5)
    checkpoint = torch.load(ckp_path)
    resume_epoch = checkpoint['epoch']
    conv_net.load_state_dict(checkpoint['conv_state_dict'], strict=True)

    spd_net.load_state_dict(checkpoint['spd_state_dict'], strict=True)
    conv_net = conv_net.cuda()
    spd_net = spd_net.cuda()
    conv_net = conv_net.eval()
    spd_net = spd_net.eval()


train_img_loader = VideoLoader_from_lst(root_dir=data_dir, lst_video=lst_subjects, transform=train_transforms,
                                        loader=default_loader)
train_loader = torch.utils.data.DataLoader(train_img_loader,
                                           batch_size=1, shuffle=False)

start_time = time.time()
cnt = 0
for i, (inputs, media_labels, video_name) in enumerate(train_loader, 0):

    if i % 10 == 0:
        print('process {} templates'.format(i))
    np_media_label = media_labels.numpy()
    np_media_label = np_media_label[0]
    ib = 0
    while True:
        start = ib * batch_size
        if start > len(inputs) - 1:
            break
        end = min((ib + 1) * batch_size, len(inputs))
        ib += 1
        cur_inputs = inputs[start:end]
        img_data = Variable(torch.cat(cur_inputs)).cuda()
        feat = []
        if net_type == 'resnet_34':
            conv_4, conv, feat, logits = net(img_data)

        if net_type == 'resnet_34_coso':
            conv_4 = conv_net(img_data)
            conv, feat, logits = spd_net(conv_4)

        cur_feat = feat.data.cpu().numpy()
        if ib == 1:
            emb = cur_feat
        else:
            emb = np.vstack((emb, cur_feat))

    sname = video_name[0]
    hf.create_dataset(sname, data=emb)
    hf.create_dataset(sname + '_media_label', data=np_media_label)
    hf.flush()

hf.close()

