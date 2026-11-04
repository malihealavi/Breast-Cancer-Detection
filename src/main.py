import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from train.Trainer import GPUMammoTrainer
from utils.utils import group_views_by_patient_laterality
from dataset.augmentation import (train_transforms , valid_transforms)

from configs import get_config
cfg = get_config()

# -----------------------------------------
#                 Execution
# -----------------------------------------

def run(train_slice, val_slice, train_transforms, valid_transforms):
    torch.manual_seed(42)       # was: xm.set_rng_state(42)
    try:
        trainer = GPUMammoTrainer(
            train_slice=train_slice, val_slice=val_slice, epochs=50,
            train_transforms=train_transforms, valid_transforms=valid_transforms
        )
        trainer.training()
    except Exception:
        import traceback
        with open("/kaggle/working/crash.log", "w") as f:
            traceback.print_exc(file=f)
        raise

if __name__ == '__main__':

    

    train_df = pd.read_csv('./assets/data/train_with_bbox.csv')
    bbox_cols = ['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max']
    train_df['bbox'] = train_df[bbox_cols].apply(
        lambda row: row.tolist() if row.notna().all() else None,
        axis=1
    )
    train_df.to_csv('/kaggle/working/train_with_bbox.csv', index=False)
    train_df['patient_id'] = train_df['patient_id'].astype(str)
    train_df['image_id'] = train_df['image_id'].astype(str)
    print(f"length of original df : {len(train_df)}")
    train_df = group_views_by_patient_laterality(train_df)
    print(f"length of df after grouping on patient_id and laterality : {len(train_df)}")

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    folds = sgkf.split(X=train_df, y=train_df['cancer'], groups=train_df['patient_id'])
    train_idx, val_idx = next(folds)
    train_slice = train_df.iloc[train_idx].reset_index(drop=True)
    val_slice = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"length train_set: {len(train_slice)} , length val_set: {len(val_slice)}")
    print(f"train value_count(cancer) : {train_slice['cancer'].value_counts()}")
    print(f" val value_count(cancer): {val_slice['cancer'].value_counts()}")

    run(train_slice, val_slice, train_transforms(cfg.glob.IMG_SIZE), valid_transforms(cfg.glob.IMG_SIZE))  