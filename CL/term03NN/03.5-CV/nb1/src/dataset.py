import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2
from pathlib import Path

def label_builder(categories, image_path):
    img_lab = pd.DataFrame(sorted(str(p) for p in image_path.glob('*.*g')), 
                           columns=['name'])
    img_lab['label'] = np.nan
    for i, cat in enumerate(categories):
        img_lab.loc[img_lab['name'].str.contains(cat), 'label'] = i
    img_lab.dropna(inplace=True)
    img_lab['label'] = img_lab['label'].astype('int64')
    return img_lab


class CustomImageDataset(Dataset):
    def __init__(self, img_df, transform=None):
        super().__init__()
        self.img_labels = img_df
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image=image)
        return image, label
        
