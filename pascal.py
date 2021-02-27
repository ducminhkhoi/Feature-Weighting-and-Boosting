

import torch.utils.data as data
import os
from PIL import Image
from utils import preprocess, get_cats, AvgPool2d
import numpy as np
import matplotlib.pyplot as plt
import torch 
import random
random.seed(1991)
from random import choice
from torchvision import transforms

class VOCSegmentationRandom(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, 
                        crop_size=512, group=0, num_folds=4, num_shots=1, batch_size=8, iteration=10000):
        self.root = root
        _voc_root = os.path.join(self.root, 'VOC2012')
        _list_dir = os.path.join(_voc_root, 'list')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size

        if group == 'all':
            self.cats = range(1, 21)
        else:
            self.cats = [x + 1 for x in sorted(get_cats('train' if train else 'val', group, num_folds))]

        self.cats_set = set(self.cats)
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.cat_dict = {c:i for i, c in enumerate(self.cats)}

        if self.train:
            _list_f = os.path.join(_list_dir, 'train_aug.txt')
        else:
            _list_f = os.path.join(_list_dir, 'val.txt')

        if train:
            data_file = 'data/data_{}_{}.pkl'.format(group, num_folds)
        else:
            data_file = 'data/pascal_val_{}_{}.pkl'.format(group, num_folds)

        if not train:
            file = 'data/val_pascal_{}_{}_5shot_new.pkl'.format(group, num_folds)
            self.list_data = torch.load(file)

        if not os.path.isfile(data_file):
            self.images = {k: {} for k in self.cats}
            self.list_images = {k: [] for k in self.cats}
            with open(_list_f, 'r') as lines:
                for line in lines:
                    img_id = line.split()[0][12:23].replace('\n', '')
                    print(img_id)
                    _image = _voc_root + line.split()[0]
                    _mask = _voc_root + line.split()[1]
                    labels = np.array(Image.open(_mask))
                    img_label = set(x for x in np.unique(labels).tolist() if x not in [255, 0] and (labels==x).sum() > 1000)

                    real_label = list(img_label & self.cats_set)
                    for label in real_label:
                        self.images[label][img_id] = (_image, _mask, real_label)

                        self.list_images[label].append(img_id)

            torch.save((self.images, self.list_images), data_file)
            print('finished')
        else:
            self.images, self.list_images = torch.load(data_file)

        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=crop_size//8, radius=5)
        self.iteration = iteration

    def __getitem__(self, index):
        if self.train:
            chosen_label = choice(self.cats)
            image_id_q = choice(self.list_images[chosen_label])
            image_id_s = choice(self.list_images[chosen_label])
        else:
            image_id_q, image_id_s, chosen_label = self.list_data[index]

        img_q, target_q = self.get_img_info(chosen_label, image_id_q)

        if self.num_shots == 1:
            img_s, target_s = self.get_img_info(chosen_label, image_id_s)
            return img_q, target_q, img_s, target_s, self.cat_dict[chosen_label]
        else:
            img_s, target_s = [], []
            for image_id_ in image_id_s:
                img_, target_ = self.get_img_info(chosen_label, image_id_)
                img_s.append(img_)
                target_s.append(target_)

            img_q, img_s[0] = img_s[0], img_q
            target_q, target_s[0] = target_s[0], target_q

            return img_q, target_q, img_s, target_s, chosen_label
        

    def get_img_info(self, chosen_label, image_id):
        # image_id = choice(self.list_images[chosen_label])

        image = self.images[chosen_label][image_id]
        
        _img = Image.open(image[0]).convert('RGB')

        _target = Image.open(image[1])

        _img, _target = preprocess(_img, _target,
                                   flip=False,
                                   scale=None,
                                   crop=(self.crop_size, self.crop_size))

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        target = torch.zeros_like(_target)

        target[_target.int() == int(chosen_label)] = 1        

        return _img, target

    def __len__(self):
        if self.train:
            return self.iteration * self.batch_size
        else:
            return len(self.list_data)


if __name__ == '__main__':
    dataset = VOCSegmentationRandom('data/VOCdevkit', train=False)

    for data in dataset:
        print()