import os
import torch
import resnet
import torchvision.transforms as transforms
import torch.utils.data as data
from read_utils import ImageFilelist
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as Lr_Sheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils


# directory for saving models
save_dir = './data/WebFace/model/resnet34'
# training image file list
train_filelist = './data/WebFace/train_cropped_extend_no_overlap.txt'
# directory of training images
root_dir = './data/WebFace/cropped_extend'

train_batch_size = 256
input_image_size = 224
num_classes = 10575 - 25
step = 0
lr_nn = 0.2
num_epoch = 40

train_transforms = transforms.Compose([
    transforms.RandomCrop(input_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5409566, 0.41063643, 0.3478864), (0.27481332, 0.23811759, 0.22841948))
])
train_img_loader = ImageFilelist(root_dir=root_dir, fname=train_filelist, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_img_loader, batch_size=train_batch_size, shuffle=True, num_workers=8)

net = resnet.resnet34(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()

net = net.cuda()
criterion = criterion.cuda()


optimizer_nn = optim.SGD(net.parameters(), lr=lr_nn, momentum=0.9, weight_decay=5e-4)
lr_sheduler_nn = Lr_Sheduler.MultiStepLR(optimizer_nn, milestones=[0, 5, 10, 15, 21, 27, 32, 35], gamma=0.5)
hist_loss = []
hist_acc = []

def train(epoch, step):
    net.train()
    for i, (inputs, targets, fnames) in enumerate(train_loader, 0):

        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()

        conv, pre_feat, logits = net(inputs)
        loss = criterion(logits, targets)

        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()

        # train_loss += loss.data[0]
        _, predicted = torch.max(logits.data, 1)
        accuracy = (targets.data == predicted).float().mean()
        if step % 10 == 0:
            np_loss = loss.data[0]
            np_acc = accuracy
            print('[epoch %d step %d] loss %f acc %f' % (epoch, step, np_loss, np_acc))

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
                'net_state_dict': net.state_dict(),
            }, filename=os.path.join(save_dir, 'step_' + str(step) + '.model'))

        step += 1
    return step


start_epoch = 0
resume_epoch = start_epoch
if step > 0:
    rpath = os.path.join(save_dir, 'step_' + str(step) + '.model')
    checkpoint = torch.load(rpath)
    resume_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net_state_dict'])
    print('resume from {}'.format(rpath))
    # optimizer.load_state_dict(checkpoint['optimizer'])

start_epoch = resume_epoch
for epoch in range(start_epoch, start_epoch + num_epoch):
    lr_sheduler_nn.step()
    print('\nEpoch: %d of %d' % (epoch, num_epoch + start_epoch))
    print('learning rate')
    for param_group in optimizer_nn.param_groups:
        print(param_group['lr'])

    step = train(epoch, step)
    utils.save_checkpoint({
        'epoch': epoch,
        'step': step,
        'net_state_dict': net.state_dict(),
    }, filename=os.path.join(save_dir, 'step_' + str(step) + '.model'))