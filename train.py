##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import cv2
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import argparse


# timm
from PIL import Image
import cv2


##
import torch
import torch.nn.functional as F

### etc ###
from tqdm import tqdm
from sklearn.manifold import TSNE
from glob import glob
#time
import time



#args
import argparse



#### 모듈들
from modules.loader import data_split
from modules.dataset import ColonoscopyDataset
from modules.model import model_loader
from modules.trainer import train
from modules.eval import test

####


def main(config):
    
    # 폴더 생성
    # checkpoint, results
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    if not os.path.exists('./results'):
        os.makedirs('./results')

    
    # 1. data split
    
    train_list, train_mask_list, valid_list, test_list, valid_mask_list, test_mask_list = data_split(config)
    
    # 데이터셋 수 확인.
    print("train_list : ", len(train_list))
    print("valid_list : ", len(valid_list))
    print("test_list : ", len(test_list))
    
    
    #########
    # 2. dataset 제작
    img_transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((352, 352),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    train_dataset = ColonoscopyDataset(train_list, train_mask_list, transform=img_transform, mask_transform=mask_transform)
    valid_dataset = ColonoscopyDataset(valid_list, valid_mask_list, transform=img_transform, mask_transform=mask_transform)
    test_dataset = ColonoscopyDataset(test_list, test_mask_list, transform=img_transform, mask_transform=mask_transform)        
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,num_workers=config.num_workers , shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size,num_workers=config.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)    
    
    # train dataset 이미지 확인
    for img, mask in train_dataset:
        print(img.shape, mask.shape)
        break
    # 이미지 저장
    # visualize
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.imshow(mask[0], cmap='gray')

    #둘의 곱
    plt.subplot(1, 3,3)
    plt.imshow( (img * mask[0]).permute(1, 2, 0), cmap='gray')
    plt.show()
    # save fig
    plt.savefig('./results/sample.png')
    ###########


    # 3. model
    model = model_loader(config)
    model = model.cuda()    
    
    # 4. train code
    model, train_loss, valid_loss = train(config, model, train_loader, valid_loader)

    
    
    # loss visualize
    # visualize
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.legend()
    plt.show()
    # save
    plt.savefig('./results/loss.png')

    # 5. eval
    testset_list = ["In","CVC-300", "CVC-ClinicDB","ETIS-LaribPolypDB","Kvasir"]
    
    for testset in testset_list:
        print("----")
        print(testset)
        test_folder_path = f"{config.test_folder_path}/{testset}"
        
        if testset == "In":
            pass
        else:
            test_img_path = os.path.join(test_folder_path, 'images') + '/*.png'
            test_mask_path = os.path.join(test_folder_path, 'masks') + '/*.png'
            test_list = glob(test_img_path)
            test_mask_list = glob(test_mask_path)
        
        test_dataset = ColonoscopyDataset(test_list, test_mask_list, transform=img_transform, mask_transform=mask_transform)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        len_testset = len(test_mask_list)
        print("testset length : ", len_testset)
        
        test(model, test_loader, len_testset)
    
    return

if __name__ == '__main__':
    
    # parse args
    args = argparse.ArgumentParser()
    
    args.add_argument('--model', type=str, default='hybrid', help='[unet,pvt,resnet,hybrid]')
    
    args.add_argument('--train_folder_path', type=str, default='./data/TrainDataset')
    args.add_argument('--test_folder_path', type=str, default='./data/TestDataset')
    
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_epochs', type=int, default=100)
    args.add_argument('--learning_rate', type=float, default=0.0001)
    args.add_argument('--num_workers', type=int, default=0)
    config = args.parse_args()
        
    # BATCH_SIZE = 8
    # NUM_EPOCHS = 100
    # LEARNING_RATE = 0.0001

    
    main(config)
    