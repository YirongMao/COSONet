import os
import torch
import resnet
import torch.utils.data as data
from read_utils import ImageFilelist
import read_utils
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as Lr_Sheduler
import time
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
                                description='training hyper parameters')


parser.add_argument('--gpus',
                    type=str,
                    default='0',
                    help='')
parser.add_argument('--train_batch_size',
                    type=int,
                    default=128,
                    help='batch size')
parser.add_argument('--lr_nn', type=float, default=0.001,
                    help='learning rate for the pre-trained first 4 convolution blocks')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate for newly layers')
parser.add_argument('--num_epoch', type=int, default=20,
                    help='')
parser.add_argument('--step', type=int, default=0,
                    help='current step')
parser.add_argument('--save_dir', type=str, default='./data/WebFace/model/resnet34_coso',
                    help='directory for saving models')

parser.add_argument('--train_filelist', type=str, default='./data/WebFace/train_cropped_extend_no_overlap.txt',
                    help='training image file list')
parser.add_argument('--img_root_dir', type=str, default='./data/WebFace/cropped_extend',
                    help='training image root directory')
parser.add_argument('--pre_trained_model', type=str, default='./data/WebFace/model/resnet34/step_118860.model',
                    help='path of pre-trained model')
parsed = parser.parse_args()

lr_nn = parsed.lr_nn
lr = parsed.lr
num_epoch = parsed.num_epoch
save_dir = parsed.save_dir
train_filelist =parsed.train_filelist
root_dir = parsed.img_root_dir
pre_trained_model = parsed.pre_trained_model
gpus = parsed.gpus
train_batch_size = parsed.train_batch_size
step = parsed.step

input_image_size = 224
num_classes = 10575 - 25
lst_gpu = list(gpus.split(','))
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = utils.config_log(log_dir=save_dir, fname='webface_resnet34_coso_train')
logger.info(parsed)

# image loader
train_transforms = read_utils.webface_train_transforms
train_img_loader = ImageFilelist(root_dir=root_dir, fname=train_filelist, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_img_loader, batch_size=train_batch_size, shuffle=True, num_workers=12)

# network
conv_net, spd_net = resnet.resnet34_cov(num_classes=num_classes, norm_type='T', num_iter=5)
criterion = torch.nn.CrossEntropyLoss()
conv_net = conv_net.cuda()
spd_net = spd_net.cuda()
criterion = criterion.cuda()
checkpoint = torch.load(pre_trained_model)
conv_net.load_state_dict(checkpoint['net_state_dict'])

# optimizer
optimizer = optim.SGD(spd_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_nn = optim.SGD(conv_net.parameters(), lr=lr_nn, momentum=0.9, weight_decay=5e-4)
lr_sheduler = Lr_Sheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)
lr_sheduler_nn = Lr_Sheduler.MultiStepLR(optimizer_nn, milestones=[4, 8, 12, 16], gamma=0.5)

hist_loss = []
hist_acc = []
def train(epoch, step):
    conv_net.train()
    spd_net.train()
    start_time = time.time()
    for i, (inputs, targets, fnames) in enumerate(train_loader, 0):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        conv_4 = conv_net(inputs)
        conv_5, feat, logits = spd_net(conv_4)
        loss = criterion(logits, targets)

        optimizer_nn.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer_nn.step()
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        accuracy = (targets.data == predicted).float().mean()

        if step % 501 == 0:
            logger.info('learning rate')
            for param_group in optimizer_nn.param_groups:
                logger.info(param_group['lr'])
            for param_group in optimizer.param_groups:
                logger.info(param_group['lr'])

        if step % 10 == 0:
            np_loss = loss.data[0]
            np_acc = accuracy
            logger.info(
                '[epoch %d step %d] loss %f acc %f time %f' % (epoch, step, np_loss, np_acc, time.time() - start_time))
            start_time = time.time()
            hist_loss.append(np_loss)
            hist_acc.append(np_acc)

            plt.subplot(211)
            plt.plot(list(range(len(hist_loss))), hist_loss)
            plt.title('Loss')
            plt.grid(True)

            plt.subplot(212)
            plt.plot(list(range(len(hist_acc))), hist_acc)
            plt.title('Accuracy')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_acc_1.jpg'))
            plt.close()

        if step % 2000 == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'step': step,
                'conv_state_dict': conv_net.state_dict(),
                'spd_state_dict': spd_net.state_dict(),
                'hist_loss': hist_loss,
                'hist_acc': hist_acc,
            }, filename=os.path.join(save_dir, 'step_' + str(step) + '.model'))

        step += 1
    return step


start_epoch = 0
resume_epoch = start_epoch
if step > 0:
    checkpoint = torch.load(os.path.join(save_dir, 'step_' + str(step) + '.model'))
    resume_epoch = checkpoint['epoch']
    conv_net.load_state_dict(checkpoint['conv_state_dict'])
    spd_net.load_state_dict(checkpoint['spd_state_dict'])
    hist_loss = checkpoint['hist_loss']
    hist_acc = checkpoint['hist_acc']
    logger.info('resume from {}'.format(os.path.join(save_dir, 'step_' + str(step) + '.model')))

start_epoch = resume_epoch
for epoch in range(start_epoch, start_epoch + num_epoch):
    lr_sheduler.step()
    lr_sheduler_nn.step()
    print('\nEpoch: %d of %d' % (epoch, num_epoch + start_epoch))
    print('learning rate')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    for param_group in optimizer_nn.param_groups:
        print(param_group['lr'])

    step = train(epoch, step)
    utils.save_checkpoint({
        'epoch': epoch,
        'step': step,
        'conv_state_dict': conv_net.state_dict(),
        'spd_state_dict': spd_net.state_dict(),
        'hist_loss': hist_loss,
        'hist_acc': hist_acc,
    }, filename=os.path.join(save_dir, 'step_' + str(step) + '.model'))