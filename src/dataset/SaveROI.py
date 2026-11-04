# !pip install --no-index --find-links=/kaggle/input/datasets/malihealavi/pydicom-packages-rsna-breast-cancer pydicom pylibjpeg pylibjpeg-libjpeg

import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

import torch
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader

from dataset.preprocessing import (normalizer , get_anatomy_bbox)

try:
    import dicomsdl
except ImportError:
    dicomsdl = None
    import pydicom
    
class DataROI(Dataset):
    def __init__(self, df: pd.DataFrame, random_crop_roi: bool = True):
        self.random_crop_roi = random_crop_roi
        self.img_dir = (
            "/kaggle/input/competitions/rsna-breast-cancer-detection/train_images"
        )
        self.records = df.to_dict(orient='records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        patient_id = row['patient_id']
        image_ids = row['image_id']

        images, view_ids = [], []
        try:
            img_id = image_ids
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
            bbox = get_anatomy_bbox(image, threshold_value=15)
            
            return bbox , patient_id , image_ids
        except Exception as e:
            print(f"[warn] skipping bad image {img_path}: {e}")

train_df = pd.read_csv('./assets/data/train.csv')
train_ds = DataROI(train_df, random_crop_roi=True)

cv2.setNumThreads(0)
os.environ['OMP_NUM_THREADS'] = '1'

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch_data.dataloader.default_collate(batch)

train_loader = DataLoader(
    dataset=train_ds, batch_size=32, num_workers=2,
    drop_last=False, timeout=300, collate_fn=collate_skip_none,
)

bbox_map = {}  # (patient_id, image_id) -> [x_min, y_min, x_max, y_max]

for batch in tqdm(train_loader, total=len(train_loader)):
    if batch is None:
        continue
    bb, patient_id, image_ids = batch
    x_min, y_min, x_max, y_max = bb

    patient_id = patient_id.tolist() if torch.is_tensor(patient_id) else list(patient_id)
    image_ids = image_ids.tolist() if torch.is_tensor(image_ids) else list(image_ids)

    for i in range(len(patient_id)):
        pid, iid = patient_id[i], image_ids[i]
        bbox_map[(pid, iid)] = [int(x_min[i]), int(y_min[i]), int(x_max[i]), int(y_max[i])]

# key on (patient_id, image_id) rather than position -> robust to dropped/None samples
train_df['bbox'] = train_df.apply(
    lambda r: bbox_map.get((r['patient_id'], r['image_id'])), axis=1
)

n_missing = train_df['bbox'].isna().sum()
print(f'{n_missing} / {len(train_df)} rows missing bbox (failed decode)')

# pickle preserves the list dtype; CSV is flattened for readability
train_df.to_pickle('./assets/data/train_with_bbox.pkl')

bbox_cols = pd.DataFrame(
    train_df['bbox'].apply(lambda b: b if isinstance(b, list) else [None]*4).tolist(),
    columns=['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max'],
    index=train_df.index,
)
train_df.drop(columns='bbox').join(bbox_cols).to_csv('./assets/data/train_with_bbox.csv', index=False)
