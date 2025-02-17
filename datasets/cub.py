import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_tencrops_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize(mean, std)
                 (transforms.ToTensor()(crop)) for crop in crops])),
    ])
    return train_transform, test_transform, test_tencrops_transform


class CUBDataset(Dataset):
    """ 'CUB <http://www.vision.caltech.edu/visipedia/CUB-200.html>'

    Args:
        root (string): Root directory of dataset where directory "CUB_200_2011" exists.
        is_train (bool): If True. create dataset from training set, otherwise creates from test set.
    """
    def __init__(self, root, is_train, is_gen_bbox):

        self.root = root
        self.is_train = is_train
        self.is_gen_bbox = is_gen_bbox
        self.resize_size = 256
        self.crop_size = 224

        self.image_list = self.remove_1st_column(open(
            os.path.join(root, 'images.txt'), 'r').readlines())
        # self.image_list是相同
        self.label_list = self.remove_1st_column(open(
            os.path.join(root, 'image_class_labels.txt'), 'r').readlines())
        self.split_list = self.remove_1st_column(open(
            os.path.join(root, 'train_test_split.txt'), 'r').readlines())
        self.bbox_list = self.remove_1st_column(open(
            os.path.join(root, 'bounding_boxes.txt'), 'r').readlines())

        self.train_transform, self.onecrop_transform, self.tencrops_transform = get_transforms()
        # if cfg.TEST.TEN_CROPS:
        #     self.test_transform = self.tencrops_transform
        # else:
        #     self.test_transform = self.onecrop_transform
        self.test_transform = self.onecrop_transform


        if is_train:
            self.index_list = self.get_index(self.split_list, '1')
        else:
            self.index_list = self.get_index(self.split_list, '0')

    def get_index(self, list, value):
        index = []
        for i in range(len(list)):
            if list[i] == value:
                index.append(i)
        return index

    def remove_1st_column(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            if len(input_list[i][:-1].split(' '))==2:
                output_list.append(input_list[i][:-1].split(' ')[1])
            else:
                output_list.append(input_list[i][:-1].split(' ')[1:])
        return output_list

    def __getitem__(self, idx):

        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root, 'images', name)

        image = Image.open(image_path).convert('RGB')
        image_size = list(image.size)
        label = int(self.label_list[self.index_list[idx]])-1


        if self.is_train:
            image = self.train_transform(image)
            return image, label
        else:
            ori_img = torch.from_numpy(np.array(image))
            image = self.test_transform(image)

            bbox = self.bbox_list[self.index_list[idx]]
            # print(f'bbox: {type(bbox)}')
            # print(f'bbox: {bbox}')

            bbox = [int(float(value)) for value in bbox]
            ori_bbox = bbox
            # print(f'ori_bbox: {type(ori_bbox)}')
            # print(f'ori_bbox: {ori_bbox}')
            # a = []
            # b = a[1]
            [x, y, bbox_width, bbox_height] = bbox
            # if self.is_train:
            #     resize_size = self.resize_size
            #     crop_size = self.crop_size
            #     shift_size = (resize_size - crop_size) // 2
            # resize_size = self.crop_size  error?
            resize_size = self.resize_size
            crop_size = self.crop_size
            shift_size = 0
            [image_width, image_height] = image_size
            left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))
            right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
            right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))

            # gt_bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            # gt_bbox = torch.tensor(gt_bbox)
            gt_bbox = np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            # print(f'gt_bbox: {gt_bbox}')
        if self.is_gen_bbox:
            gt_bbox = " ".join(list(map(str, ori_bbox)))
            return ori_img, image, label, gt_bbox, name
            # print(f'ori_bbox: {gt_bbox}')
        return image, label, gt_bbox, name


    def __len__(self):
        return len(self.index_list)








