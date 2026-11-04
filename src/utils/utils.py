import torch
from torch import nn
import numpy as np
import os 
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import math
from configs import get_config
cfg = get_config()

def freeze_bn(model):
    for m in model.encoder.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
# -----------------------------------------
#    SubSample Train with original ratio
# -----------------------------------------

def build_epoch_shards(df, label_col='cancer', n_shards=5, seed=42):
   
    skf = StratifiedKFold(n_splits=n_shards, shuffle=True, random_state=seed)
    return {i: df.index[test_idx].tolist()
            for i, (_, test_idx) in enumerate(skf.split(df, df[label_col]))}

def get_epoch_subset(df, n_shards, epoch, base_seed=42):
    cycle, shard_idx = divmod(epoch, n_shards)
    shard_map = build_epoch_shards(df, n_shards=n_shards, seed=base_seed + cycle)
    return df.loc[shard_map[shard_idx]].reset_index(drop=True)


def create_balanced_patient_subset(df, neg_sampler, ratio):
    patient_targets = df.groupby('patient_id')['cancer'].max()
    pos_patient_ids = sorted(patient_targets[patient_targets == 1].index.tolist())
    n_neg_needed = len(pos_patient_ids) * ratio
    sampled_neg_patient_ids = neg_sampler.sample(n_neg_needed)

    balanced_patient_ids = pos_patient_ids + sampled_neg_patient_ids
    balanced_df = df[df['patient_id'].isin(balanced_patient_ids)].copy()
    return balanced_df

def group_views_by_patient_laterality(df):
    df = df.copy()
    df['_view_rank'] = df['view'].map(cfg.glob.VIEW_TYPE_MAP).fillna(6)
    df = df.sort_values(['patient_id', 'laterality', '_view_rank', 'image_id'])
    return df.groupby(['patient_id', 'laterality'], sort=False).agg({
        'image_id': list,
        'view': list,
        'bbox': list,
        'cancer': 'max',
    }).reset_index()
# -----------------------------------------
#             Seed Everything
# -----------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False   
        torch.backends.cudnn.benchmark = True      

seed_everything()

# -----------------------------------------
#            Logit Adjustment
# -----------------------------------------
def logit_adjustment(raw_logits, p_train, p_real, pos_weight, eps=1e-8):
    """ 
       p_train      # empirical positive fraction in your actual downsampled training data
       p_real       # true prevalence
       pos_weight   # whatever value you passed to BCEWithLogitsLoss
       
    """
    train_odds = pos_weight * (p_train / (1 - p_train + eps))
    real_odds = p_real / (1 - p_real + eps)
    adjust = math.log(real_odds + eps) - math.log(train_odds + eps)
    return raw_logits + adjust   # still logits — sigmoid applied by caller

# -----------------------------------------
#                 Eval Metric 
# -----------------------------------------

def optimal_prob_f1_score(y_true, y_pred, beta=1):
    y_true = np.array(y_true).astype(np.float32)
    y_pred = np.array(y_pred).astype(np.float32)
    
    # Clip predictions to valid probability range, same as pfbeta
    y_pred = np.clip(y_pred, 0, 1)
    
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)
    return f1

def f2_best_threshold(all_targets, all_probs):
    """Finds the threshold that maximizes the F2 score (favors recall)."""
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    p = precisions[:-1]
    r = recalls[:-1]
    
    # Safely calculate F2 scores for all thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        f2_scores = (5 * p * r) / ((4 * p) + r)
        f2_scores = np.nan_to_num(f2_scores)

    best_f2_idx = np.argmax(f2_scores)
    best_threshold_f2 = thresholds[best_f2_idx] # Fixed NameError
    
    return best_threshold_f2, f2_scores[best_f2_idx]


def f1_best_threshold(all_targets, all_probs):
    """Finds the threshold that maximizes the F1 score (balances precision & recall)."""
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    p = precisions[:-1]
    r = recalls[:-1]
    
    # Safely calculate F1 scores for all thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = (2 * p * r) / (p + r)
        f1_scores = np.nan_to_num(f1_scores)

    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_f1_idx]
    
    return best_threshold_f1, f1_scores[best_f1_idx]
# -----------------------------------------
#               plot loss 
# -----------------------------------------

def plot_loss(val_losses, train_losses, val_f1 , val_f2, path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', linewidth=2)
    plt.plot(epochs, val_f1, label='Valid F1', color='green', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='s', linewidth=2)
    plt.plot(epochs, val_f2, label='Valid F2', color='yellow', marker='s', linewidth=2)
    
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss/Eval', fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = path
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"Plot successfully saved to {os.path.abspath(plot_path)}")


# -----------------------------------------
#                 Helper
# -----------------------------------------
def unwrap(m):
    """DataParallel stores the real model under `.module` — use this
    whenever you need to reach a custom attribute (e.g. .backbone)."""
    return m.module if isinstance(m, nn.DataParallel) else m
    
def to_scalar(x):
    """Works for both tensor and plain number."""
    return x.item() if torch.is_tensor(x) else float(x)

def _to_cpu(obj):
    """Recursively move any torch.Tensor in a nested dict to CPU.
    Deliberately does NOT recurse into list/tuple/bytes."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj

def move_batch(batch, device):
    """NEW: replaces ParallelLoader (which moved batches to the TPU for us)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}