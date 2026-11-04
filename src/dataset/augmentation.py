
import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms(img_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                           border_mode=0, value=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(95, 105), p=0.2),
        A.GaussNoise(var_limit=(1, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        A.Resize(height=2 * img_size, width=img_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

def valid_transforms(img_size):
    return A.Compose([
        A.Resize(height=2 * img_size, width=img_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])