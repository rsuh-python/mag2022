import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def base_augs(size, extra_augs=[A.NoOp()]):
    return A.Compose([
        A.SmallestMaxSize(size),
        A.CenterCrop(size, size)] + extra_augs + [
        A.Normalize((0.5, 0.5, 0.5),(1, 1, 1)),
        ToTensorV2()
    ])

def train_augs(size):
    return base_augs(
        size, 
        [A.OneOf([
            A.CoarseDropout(max_holes=26, max_height=20, max_width=20, fill_value=0, p=1),
            A.HorizontalFlip(p=1),
            A.OpticalDistortion(p=1)
            ], p=1),
         A.OneOf([
             A.GaussNoise(p=1),
             A.Blur(p=1),
             A.RandomGamma(p=1)
             ], p=1)
         ])

def valid_augs(size):
    return base_augs(size)


class AugmentedDataset(Dataset):
    
    def __init__(self, dataset, aug):
        self.dataset = dataset
        self.aug = aug

    def __len__(self):
        return len(self.dataset)   
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_new = self.aug(image=image)['image']
        return image_new, label
