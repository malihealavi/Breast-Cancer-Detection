
from dataclasses import dataclass, field
import os
import torch
import pandas as pd

def _get_default_device() -> str:
    """Returns 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DataPaths:
    save_chkp = './assets/checkpoint'
    BBOX_ROI = './assets/data/'
    
@dataclass
class GlobalVar:
    IMG_SIZE : int= 512 
    MAX_IMAGES: int = 2 # Number of views per laterlaity per patient
    NEG_TO_POS_RATIO : int= 5
    N_SHARDS : int = 3
    VIEW_TYPE_MAP = {'CC': 0, 'MLO': 1,'ML': 2 ,'LM': 3, 'AT': 4 ,'LMO': 5}  # extend / remap if other view strings (ML, LM, AT...) appear in your data
    
@dataclass
class ModelConfig:
    POOL_TYPE: str = 'max'  # 'max' | 'mean' | 'gem'
    ENCODER_BB = 'convnext_small'


@dataclass
class TrainConfig:
    SEED: int = 42
    LR_HEAD: float = 1e-4
    LR_BB: float = 1e-5
    WEIGHT_DECAY: int = 5e-3
    ACCUMULATION_STEP: int  = 4
    BATCH_SIZE: int = 16
    WARMUP_EPOCHS: int  = 4
    PATIENCE: int  = 8
    USE_GRAD_CHECKPOINTING = True   # set False to turn it off
    USE_AMP = True                 # True = mixed precision (faster, less memory)
    AMP_DTYPE = torch.float16       # T4 / P100 / V100: float16 (uses GradScaler)
    NUM_WORKERS: int = 4   


@dataclass
class InferenceConfig:
    IMG_SIZE : int= 1024 
    BATCH_SIZE = 16


@dataclass
class Config:
    data: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    glob : GlobalVar = field(default_factory=GlobalVar)
    device: str = field(default_factory=_get_default_device)

def get_config() -> Config:
    return Config()