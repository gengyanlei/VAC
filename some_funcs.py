'''
    测试时使用的一些函数集合，便于代码整理
'''
import os
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from torchvision import transforms

def data_process1(img_path, resize_size=(512,512)):
    '''
    数据增强：先采用padding，使图片变成正方形，然后再数据预处理
    :param img_path:  路径
    :param resize_size:  形变大小
    :return:
    '''
    img = cv2.imread(img_path, -1)  # bgr
    assert len(img.shape) == 3
    H, W, C = img.shape
    max_length = max(H, W)
    top_size, bottom_size, left_size, right_size = (int((max_length - H) / 2), int(np.ceil((max_length - H) / 2)), int((max_length - W) / 2),int(np.ceil((max_length - W) / 2)))
    img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    H_, W_, _ = img.shape
    assert H_ == W_, '长宽比未变成1'

    img = img[..., ::-1]  # bgr->rgb
    img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize(size=resize_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img

def data_process(img_path, resize_size=(512,512)):
    '''
    普通的数据预处理
    :param img_path:
    :param resize_size:
    :return:
    '''
    img = Image.open(img_path).convert('RGB')
    img = transforms.Resize(size=resize_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    return img

def paint_chinese_opencv(img, chinese, position, fillColor):
    '''
    在图片上显示中文
    :param img: 输入图像矩阵，cv2.imread
    :param chinese: 显示中文/英文也可以
    :param position: 文字在图像上的位置
    :param fillColor: 颜色
    :return:
    '''
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', 25)
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def check_path(path):
    '''
    检查路径是否存在，不存在自动创建
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return






