


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from glob import glob


import os

def data_split(config):
    # dataset load. split
    # testset load

    train_img_path = os.path.join(config.train_folder_path, 'image') + '/*.png'
    train_mask_path = os.path.join(config.train_folder_path, 'mask') + '/*.png'

    
    train_list = glob(train_img_path)
    train_mask_list = glob(train_mask_path)
    
    # 정렬
    train_list.sort()
    train_mask_list.sort()
    
    
    # train valid split
    # shuffle (동영상이기에 섞으면, 연속된 프레임이 섞이게 됨. 향후 문제가 될 수 있긴하다.)
    # train, test, valid split 8:1:1

    train_list, rest_list, train_mask_list, rest_mask_list = train_test_split(train_list, train_mask_list, test_size=0.2, random_state=42)

    valid_list, test_list, valid_mask_list, test_mask_list = train_test_split(rest_list, rest_mask_list, test_size=0.5, random_state=42)
    
    return train_list, train_mask_list, valid_list, test_list, valid_mask_list, test_mask_list

        
