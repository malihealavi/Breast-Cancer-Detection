import torch
from torch import nn
from models.ImageEncoder import ImageEncoderPerView
from models.ViewSetPooling import ViewSetPooling

class MammoModelHybrid(nn.Module):
    def __init__(self, model_name='efficientnetv2_rw_m'):
        super().__init__()
        self.encoder = ImageEncoderPerView(model_name)
        # self.aggregator = ViewSetAggregator(feat_dim=self.encoder.num_features)
        self.aggregator = ViewSetPooling(feat_dim=self.encoder.num_features, pool_type='max') 
    def forward(self, images, view_type_ids, valid_mask):
        feats = self.encoder(images)
        return self.aggregator(feats, view_type_ids, valid_mask)
