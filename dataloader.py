import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import random
import numpy as np
from PIL import Image


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Random_PatchShuffle(object):
    def __init__(self, patch_h, patch_w, random_index):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.random_index = random_index
    def __call__(self, img):
        random_temp = random.uniform(0,1)
        if random_temp > self.random_index:
            img_np = np.asarray(img)
            img_gen = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
            img_h = img.size[1]
            img_w = img.size[0]
            num_h = img_h // self.patch_h
            num_w = img_w // self.patch_w
            num_all = num_h * num_w
            random_list = np.random.permutation(num_all)
            index = 0
            for i in random_list:
                temp_h = i // num_w
                temp_w = i % num_w
                index_h = index // num_w
                index_w = index % num_w
                img_gen[self.patch_h*temp_h:self.patch_h*(temp_h+1), self.patch_w*temp_w:self.patch_w*(temp_w+1), :] = \
                    img_np[self.patch_h*index_h:self.patch_h*(index_h+1), self.patch_w*index_w:self.patch_w*(index_w+1), :]
                index += 1
            img_output = Image.fromarray(img_gen)
        else:
            img_output = img

        return img_output

def get_dataloaders(args):
    train_transform = transforms.Compose([
        Random_PatchShuffle(200,300,0.7),
        transforms.Resize((200,300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=1, contrast=1, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # Cutout(n_holes=1, length=2500)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((200,300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')
    train_set = datasets.ImageFolder(traindir, train_transform)
    val_set = datasets.ImageFolder(valdir, val_transform)

    if 'train' in args.splits:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    if 'val' or 'test' in args.splits:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        test_loader = val_loader
        
    return train_loader, val_loader, test_loader

# -----------------------------------------------------------------------------------------------------------------------------------
def get_dataloaders_bak(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
