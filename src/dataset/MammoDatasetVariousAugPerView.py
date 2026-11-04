import torch
from torch.utils.data import Dataset, DataLoader
import os 
import pandas as pd
import numpy as np
try:
    import dicomsdl
except ImportError:
    dicomsdl = None
    import pydicom
import albumentations as A


from dataset.preprocessing import (
    get_anatomy_bbox,
    get_roi_random_crop, 
    normalizer
)
from configs import get_config
cfg = get_config()

class MammoDatasetVariousAugPerView(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None, split: str = 'train',
                 max_images: int = cfg.glob.MAX_IMAGES, random_crop_roi: bool = True):
        self.transforms = transforms
        self.split = split
        self.max_images = max_images
        self.random_crop_roi = random_crop_roi
        self.img_dir = (
            "/kaggle/input/competitions/rsna-breast-cancer-detection/train_images"
            if split == 'train' else
            "/kaggle/input/competitions/rsna-breast-cancer-detection/test_images"
        )
        self.records = df.to_dict(orient='records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        patient_id = row['patient_id']
        image_ids = row['image_id']

        # ASSUMPTION: row['view'] is a list parallel to image_id, e.g. ['CC','MLO'].
        # If your df instead stores a single scalar 'view' per row, this breaks — confirm structure.
        views_raw = row.get('view', None)
        if views_raw is None:
            views_raw = ['CC'] * len(image_ids)
        elif not isinstance(views_raw, (list, tuple, np.ndarray)):
            views_raw = [views_raw]

        n_real = min(len(image_ids), self.max_images)
        images, view_ids = [], []

        for i in range(n_real):
            try:
                img_id = image_ids[i]
                img_path = os.path.join(self.img_dir, str(patient_id), f"{img_id}.dcm")
                try:
                    dcm = dicomsdl.open(img_path)
                    image = dcm.pixelData()
                    photometric = dcm.PhotometricInterpretation
                    try:
                        dcm_meta = pydicom.dcmread(img_path, stop_before_pixels=True)
                    except Exception:
                        dcm_meta = None
                except Exception:
                    try:
                        dcm = pydicom.dcmread(img_path)
                        image = dcm.pixel_array
                        photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
                        dcm_meta = dcm
                    except FileNotFoundError:
                        print(f'FileNotFoundError: {img_path}')
                        raise
    
                image = normalizer(image, photometric, dcm_meta=dcm_meta)
                bbox = row['bbox'][i]
                if bbox is None:
                    bbox = get_anatomy_bbox(image, threshold_value=15)
                tight_image = get_roi_random_crop(
                    image, bbox, crop_h=2*cfg.glob.IMG_SIZE, crop_w=cfg.glob.IMG_SIZE,
                    roi_padding=10, random_position=self.random_crop_roi
                )
            except Exception as e:
                print(f"[warn] skipping bad image {img_path}: {e}")
                image = np.zeros((2*cfg.glob.IMG_SIZE, cfg.glob.IMG_SIZE), dtype=np.uint8)
                tight_image = image
                
            if self.transforms:
                augmented = self.transforms(image=tight_image)
                image = augmented['image']
                if not isinstance(image, torch.Tensor):
                    image = torch.tensor(image, dtype=torch.float32) / 255.0
                    if image.ndim == 2:
                        image = image.unsqueeze(0)
            else:
                image = tight_image.astype(np.float32) / 255.0
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            images.append(image)                                   # (1, H, W)
            view_str = views_raw[i] if i < len(views_raw) else 'CC'
            view_ids.append(cfg.glob.VIEW_TYPE_MAP.get(view_str, 0))

        # --- Pad: cycle-repeat real (image, view_id) pairs, no zeros, no re-decode ---
        valid_mask = [1] * n_real + [0] * (self.max_images - n_real)
        for i in range(n_real, self.max_images):
            if n_real > 0:
                src = i % n_real
                images.append(images[src].clone())
                view_ids.append(view_ids[src])
            else:
                # Rare: breast side with zero images at all
                images.append(torch.zeros((1, 2*cfg.glob.IMG_SIZE, cfg.glob.IMG_SIZE), dtype=torch.float32))
                view_ids.append(0)

        images_tensor = torch.stack(images, dim=0)   # (max_images, 1, H, W) <- stack, not cat

        data_packet = {
            'image': images_tensor,
            'view_type_ids': torch.tensor(view_ids, dtype=torch.long),   # (max_images,)
            'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),    # (max_images,)
            'patient_id': patient_id,
            'laterality': row['laterality'],
        }

        # exclude_cols = {'patient_id', 'image_id', 'biopsy', 'invasive', 'density', 'laterality',
        #                 'BIRADS', 'difficult_negative_case', 'cancer', 'prediction_id', 'view'}
        # metadata = [float(row[k]) for k in row.keys() if k not in exclude_cols and pd.notna(row[k])]

        if self.split == 'train':
            data_packet['target'] = torch.tensor(row['cancer'], dtype=torch.float32)
        else:
            data_packet['prediction_id'] = str(row.get('prediction_id', f"{patient_id}_{row['laterality']}"))

        return data_packet