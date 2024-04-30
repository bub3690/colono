
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import cv2

## dataset 제작
class ColonoscopyDataset(Dataset):
    def __init__(self, img_list, mask_list, transform=None, mask_transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
        
        # 만약 마스크가 1채널이 아니라면, 1채널로 변경
        if mask.shape[0] != 1:
            # 하나로 압축. 3채널 중 하나라도 0보다 크면 1 표시.
            mask = mask[0].unsqueeze(0)
            mask[mask > 0] = 1
        
        return img, mask