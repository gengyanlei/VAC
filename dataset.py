"""
    author is leilei
"""
import cv2
import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

__all__ = ['get_subsets', 'CustomDataset']

class CustomDataset(Dataset):
    def __init__(self, phase='train', size1=(448,448), size2=(384,384)):
        assert phase in ['train', 'test']

        self.size1 = size1
        self.size2 = size2
        self.phase = phase
	''' 这里3个变量，可以通过传参进行修改！！！ '''
        self.root = '/home/ailab/dataset/new_data/'
        train_txt_file = open('/home/ailab/dataset/new_data/readme/train.txt')
        test_txt_file = open('/home/ailab/dataset/new_data/readme/test.txt')

        if self.phase == 'train':
            train_img_list = []
            train_lab_list = []
            for line in train_txt_file:
                img_name, lab = line.strip('\n').split(' ')
                lab = int(lab)
                line = os.path.join(self.root, 'image', img_name)
                train_img_list.append(line)
                train_lab_list.append(lab)

            self.data_list = train_img_list
            self.label_list = train_lab_list

        if self.phase == 'test':
            test_img_list = []
            test_lab_list = []
            for line in test_txt_file:
                img_name, lab = line.strip('\n').split(' ')
                lab = int(lab)
                line = os.path.join(self.root, 'image', img_name)
                test_img_list.append(line)
                test_lab_list.append(lab)

            self.data_list = test_img_list
            self.label_list = test_lab_list

        # transform
        if self.phase == 'train':
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            self.transform_color = transforms.Compose([
                    transforms.ColorJitter(brightness=0.3,
                                           contrast=0.3,
                                           saturation=0.2,
                                           hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=self.size1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # if self.phase == 'train':
        #     self.transform = transforms.Compose([
        #         transforms.Resize(size=(self.shape[0], self.shape[1])),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ColorJitter(brightness=0.4,
        #                                contrast=0.4,
        #                                saturation=0.4,
        #                                hue=0.2),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.Resize(size=(self.shape[0], self.shape[1])),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])

    def __edge_filling(self, img):
        '''
        以图片高宽最大的为基准，在另一个上添加padding，使得高宽比例为 "1"
        :param img: bgr,[H,W,C]
        :return: img: bgr
        '''
        assert len(img.shape) == 3
        H, W, C = img.shape
        max_length = max(H,W)
        top_size, bottom_size, left_size, right_size = (int((max_length-H)/2), int(np.ceil((max_length-H)/2)), int((max_length-W)/2), int(np.ceil((max_length-W)/2)))
        replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        H_, W_, _ = replicate.shape
        assert H_ == W_, '长宽比未变成1'

        return replicate

    def __getitem__(self, index):
        '''
        :param index:
        :return: 依次返回： 原始图像，原始图像对应的左右翻转； 缩小版，缩小版对应的左右翻转； 原始图像对应的颜色变化， 标签。
        '''
        ## 读取标签 ##
        label = self.label_list[index]
        ## 读取图片 ##
        if self.phase == 'train':
            image = cv2.imread(self.data_list[index], -1)  # bgr
            image = self.__edge_filling(image)  #  训练过程中 验证也需要保持不形变，测试更是如此。
            image = image[..., ::-1]  # bgr->rgb
            image = Image.fromarray(image, mode='RGB')
            # 加入padding之后，宽高比为1，然后经过2个不同尺度的resize

            img1 = image.resize(size=self.size1)  # 相当于原始图，以此为基准
            img2 = image.resize(size=self.size2)

            # large orig
            image_lo = self.transform(img1)
            # large flip
            image_lf = self.transform(img1.transpose(Image.FLIP_LEFT_RIGHT))
            # large Color Jitter 颜色系列
            ''' 
            必须考虑颜色系列，随机左右翻转变成固定的，随机颜色系列，不能提前数据扩充；解决思路如下：
            1.随机颜色在固定左右翻转之后，理论上关注点应该是一致的，但是为了防止 复杂组合造成的网络学习困难，尽量分开弄；
            2.随机颜色单独出来，单独对原始图像进行处理，理论上关注是一致的，降低了关注难度。 先实验这个，权重设置为1.
            '''
            image_color = self.transform_color(img1)

            # small orig
            image_so = self.transform(img2)
            # small flip
            image_sf = self.transform(img2.transpose(Image.FLIP_LEFT_RIGHT))

            return image_lo, image_lf, image_so, image_sf, image_color, label
            # return image_lo, image_lf, image_so, image_sf, label
        else:
            image = cv2.imread(self.data_list[index], -1)  # bgr
            image = image[..., ::-1]  # bgr->rgb
            image = Image.fromarray(image, mode='RGB')
            image = self.transform(image)

            return image, label

    # def __getitem__(self, index):
    #     image = Image.open(self.data_list[index]).convert('RGB')  # (C, H, W)
    #     image = self.transform(image)
    #     assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]
    #     label = self.label_list[index]
    #     return image, label

    def __len__(self):
        return len(self.label_list)

def get_subsets(size1=(448,448), size2=(384,384)):
	trainset = CustomDataset('train', size1, size2)
	testset = CustomDataset('test', size1, size2)
	return trainset, testset

if __name__ == '__main__':
    dataset = CustomDataset(phase='test')
    print(len(dataset.data_list))
    for data in dataset:
        print(data[0].shape, data[1])
        break
