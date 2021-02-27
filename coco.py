import torch.utils.data as data
import os
from PIL import Image, ImageOps
from utils import get_cats, AvgPool2d
import numpy as np
import matplotlib.pyplot as plt
import torch 
import random
random.seed(1991)
from random import choice
from torchvision import transforms
from collections import defaultdict
import json


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


def preprocess(image, mask, crop=None):
    desired_size = crop
    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = image.resize(new_size, Image.ANTIALIAS)
    pad = ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, pad)
    image = new_im

    mask = mask.resize(new_size, Image.ANTIALIAS)
    new_mask = Image.new("RGB", (desired_size, desired_size))
    new_mask.paste(mask, pad)

    mask = new_mask

    return image, mask, pad

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class COCO(data.Dataset):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False, 
                        crop_size=None, group=0, num_folds=4, batch_size=8, iteration=10000, num_shots=1):
        if root is None:
            self.root = '/home/khoinguyen/Projects/datasets/coco2017/'
        else:
            self.root = root

        self.crop_size = crop_size if train else None
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.batch_size = batch_size
        self.iteration = iteration
        self.num_shots = num_shots

        mode = 'train' if train else 'val'

        self.image_folder = self.root + '{}2017/'.format(mode)
        annotations = self.root + 'panoptic_annotations_trainval2017/'
        self.annotation_folder = annotations + 'panoptic_{}2017/'.format(mode)

        data_file = 'data/coco_{}_{}_{}.pkl'.format(mode, group, num_folds)

        if not train:
            self.list_data = torch.load('data/{}_coco_{}_{}{}.pkl'.format(mode, group, num_folds,
                                        '' if num_shots == 1 else '_5shot'))

        if not os.path.isfile(data_file): # or train:

            with open(annotations + 'panoptic_{}2017.json'.format(mode), 'r') as f:
                data = json.load(f)
            
                self.categories = [c['id'] for c in data['categories'] if c['id'] < 92]
                self.dict_cats = {c:i for i, c in enumerate(self.categories)}
                dict_cats_ = {i:c for i, c in enumerate(self.categories)}

                if group == 'all':
                    chosen_cats = set(range(80))
                else:
                    chosen_cats = set(range(group, 80, num_folds))
                    if train:
                        chosen_cats = set(range(80)) - chosen_cats

                self.categories = [dict_cats_[c] for c in chosen_cats]
                chosen_cats = set(dict_cats_[i] for i in chosen_cats)
                    
                list_images = defaultdict(set)

                self.list_categories = defaultdict(set)

                for i, anno in enumerate(data['annotations']):
                    print(i)
                    for segment in anno['segments_info']:
                        if segment['category_id'] < 92 and segment['category_id'] in chosen_cats and segment['area'] > 10000:
                            self.list_categories[anno['file_name']+'|'+str(segment['category_id'])].add(segment['id'])
                            list_images[segment['category_id']].add(anno['file_name'])

                self.list_images = {}

                print(sorted(self.categories))
                print(sorted(list(list_images.keys())))

                for k, v in list_images.items():
                    self.list_images[k] = list(v)
                    print(k, len(v))

                torch.save((self.categories, self.dict_cats, self.list_images, self.list_categories), data_file)
        else:
            self.categories, self.dict_cats, self.list_images, self.list_categories = torch.load(data_file)

    def __getitem__(self, index):
        if self.train:
            chosen_label = choice(self.categories)
            image_id_q = choice(self.list_images[chosen_label])
            image_id_s = choice(self.list_images[chosen_label])
        else:
            chosen_label, image_id_q, image_id_s = self.list_data[index]

        img_q, target_q = self.get_img_info(chosen_label, image_id_q)
        
        if self.num_shots == 1:
            img_s, target_s = self.get_img_info(chosen_label, image_id_s)
        else:
            img_s, target_s = [], []
            for image_id_ in image_id_s:
                img_, target_ = self.get_img_info(chosen_label, image_id_)
                img_s.append(img_)
                target_s.append(target_)

        return img_q, target_q, img_s, target_s, self.dict_cats[chosen_label]
        
        # chosen_label = choice(self.categories)
        # image_id_q = choice(self.list_images[chosen_label])

        # if self.num_shots == 1:
        #     image_id_s = choice(self.list_images[chosen_label])
        # else:
        #     image_id_s = [choice(self.list_images[chosen_label]) for _ in range(self.num_shots)]

        # return chosen_label, image_id_q, image_id_s

    def get_img_info(self, chosen_label, image_id):

        _img = Image.open(self.image_folder+image_id.replace('png', 'jpg')).convert('RGB')

        _target = Image.open(self.annotation_folder+image_id)

        if self.train:
            _img, _target, pad = preprocess(_img, _target, crop=self.crop_size)

        __img = data_transforms(_img)

        if self.train:
            __img[:, :pad[1]] = 0.
            __img[:, -pad[1]-1:] = 0.
            __img[:, :, :pad[0]] = 0.
            __img[:, :, -pad[0]-1:] = 0.

        _target = rgb2id(np.array(_target, dtype=np.uint8))
        _target = torch.LongTensor(np.array(_target).astype(np.int64))
        target = torch.zeros_like(_target)

        labels = self.list_categories[image_id+'|'+str(chosen_label)]
        for segment_id in labels:
            target[_target==segment_id] = 1

        return __img, target

    def __len__(self):
        if self.train:
            return self.iteration * self.batch_size
        else:
            return len(self.list_data)

if __name__ == '__main__':
    for i in [0, 1, 2, 3, 'all']:
        group = i
        coco = COCO(train=False, crop_size=512, group=group, num_shots=5)

        val_file = 'data/val_{}_{}_{}_5shot.pkl'.format('coco', group, 4)
        print(val_file)

        list_data = []
        
        for e, data in enumerate(coco):
            print(e, data)
            if e == 1000:
                break
            
            list_data.append(data)
            
        torch.save(list_data, val_file)

