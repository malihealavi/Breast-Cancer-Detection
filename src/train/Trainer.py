import warnings
warnings.filterwarnings("ignore")
import os
import time
import gc

from tqdm import tqdm
import albumentations as A

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from models.ModelHybrid import MammoModelHybrid
from dataset.BalancedPatientSampler import BalancedPatientSampler
from dataset.MammoDatasetVariousAugPerView import MammoDatasetVariousAugPerView
from utils.utils import (
    create_balanced_patient_subset,
    f1_best_threshold,
    f2_best_threshold,
    freeze_bn,
    logit_adjustment,
    move_batch,
    plot_loss,
    to_scalar,
    unwrap,
    _to_cpu

)
from configs import get_config
cfg = get_config()

# -----------------------------------------
#                 Trainer
# -----------------------------------------


class GPUMammoTrainer():
    def __init__(self, train_slice, val_slice, epochs, train_transforms, valid_transforms):
        # 1. Use the GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.accumulation_steps = cfg.train.ACCUMULATION_STEP
        self.train_slice = train_slice
        self.val_slice = val_slice
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        # 2. Instantiate Model and move to GPU
        self.model = MammoModelHybrid(model_name='efficientnetv2_rw_m')
        self.model.to(self.device)
         
        if cfg.train.USE_GRAD_CHECKPOINTING:
            self.model.encoder.backbone.set_grad_checkpointing(True)

        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     self.model = nn.DataParallel(self.model)
       
        # NEW: GradScaler is only needed for float16 (it does nothing when disabled)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(cfg.train.USE_AMP and cfg.train.AMP_DTYPE == torch.float16))

        #------------ Optimizer
        base = unwrap(self.model)
        backbone_params = base.encoder.backbone.parameters()
        head_params = base.aggregator.parameters()

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},   # small, protect pretrained features
            {'params': head_params,     'lr': 1e-4},   # needs to learn from scratch
        ], weight_decay=1e-4)

        #------------- Scheduler

        warmup = LinearLR(self.optimizer, start_factor=0.05, total_iters=cfg.train.WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.epochs - cfg.train.WARMUP_EPOCHS, eta_min=1e-8)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.WARMUP_EPOCHS])

        #------------- Loss

        self.best_f1 = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_f2_scores = []
        self.patience = 0
        self.start_epoch = 1
        self.neg_sampler_seed = 42

        #------------- predifine variables
        self.p_real = None
        self.p_train= None
        self.pos_weights= None
        self.criterion= None
        self.pr_resize = False
        self._load_checkpoint_if_exists()

    def logit_adjustment(self, raw_logits):
        return logit_adjustment(raw_logits, self.p_train, self.p_real, self.pos_weights)

    def _save_checkpoint(self, epoch, path):
        # single process: no master check, no rendezvous needed
        torch.save(_to_cpu(self._checkpoint_dict(epoch)), path)

    def _checkpoint_dict(self, epoch):
        return {
            'epoch': epoch,
            # 'model': self.model.state_dict(),            
            'model': unwrap(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'patience': self.patience,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_f2_scores': self.val_f2_scores,
            'neg_sampler_state': self.neg_sampler.state_dict(),
        }

    def _load_checkpoint_if_exists(self, path="/kaggle/input/datasets/malihealavi/breast-cancer-gpu/last_checkpoint.pth"):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting fresh.")
            return
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        unwrap(self.model).load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print('sceduler added')
        else:
            print("[warn] checkpoint has no scheduler state; scheduler starts fresh.")

        self.best_f1 = checkpoint.get('best_f1', 0.0)
        self.patience = checkpoint.get('patience', 0)
        self.start_epoch = checkpoint['epoch'] + 1

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_f1_scores = checkpoint.get('val_f1_scores', [])
        self.val_f2_scores = checkpoint.get('val_f2_scores', [])
        self.neg_sampler_state = checkpoint.get('neg_sampler_state')
        print(
            f"Resumed from {path}: last completed epoch={checkpoint['epoch']}, "
            f"best_f1={self.best_f1:.4f}, patience={self.patience}, "
            f"resuming at epoch {self.start_epoch}"
        )
        global IMG_SIZE
        if IMG_SIZE == 1024 and self.pr_resize == True:   # new, bigger resolution run
            base = unwrap(self.model)
        
            # 1. Reset the "best" values (old size scores are not comparable)
            self.best_f1 = 0.0
            self.best_val_loss = float('inf')
            self.patience = 0
            self.val_losses = []
            self.val_f1_scores = []
            self.val_f2_scores = []
        
            # 2. New optimizer with lower LR (take params from the model again)
            self.optimizer = optim.AdamW([
                {'params': base.encoder.backbone.parameters(), 'lr': 5e-6},
                {'params': base.aggregator.parameters(),       'lr': 5e-5},
            ], weight_decay=1e-4)
        
            # 3. New scheduler for the NEW optimizer (short warmup + cosine)
            remaining = self.epochs - (self.start_epoch - 1)
            warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=1)
            cosine = CosineAnnealingLR(self.optimizer, T_max=max(remaining - 1, 1), eta_min=1e-8)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[1])
        
            # 4. Fresh grad scaler (only matters for float16)
            self.scaler = torch.amp.GradScaler('cuda', enabled=(cfg.train.USE_AMP and cfg.train.AMP_DTYPE == torch.float16))
            print(
            f"Resumed from {path}: last completed epoch={checkpoint['epoch']},\n "
            f"best_f1={self.best_f1:.4f}, patience={self.patience},\n "
            f"resuming at epoch {self.start_epoch}\n"
            f"img size {IMG_SIZE}\n"
        )

    def train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0

        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for i, batch in enumerate(iterator):
            batch = move_batch(batch, self.device)
            img = batch['image']
            mask = batch['valid_mask']
            view_type_id = batch['view_type_ids']
            targets = batch['target']

            self.optimizer.zero_grad()
            outputs, _ = self.model(img, view_type_id, mask)
            loss = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()      # was: xm.optimizer_step(self.optimizer, barrier=True)

            running_loss += loss.detach() * img.size(0)

        return to_scalar(running_loss) / (len(self.train_loader.dataset) + 1e-8)


    def train_one_epoch_accumulated(self, epoch):
        self.model.train()
        freeze_bn(unwrap(self.model))
        running_loss = 0.0
        self.optimizer.zero_grad()
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for i, batch in enumerate(iterator):
            batch = move_batch(batch, self.device)
            img, mask, view_type_id = batch['image'], batch['valid_mask'], batch['view_type_ids']
            targets = batch['target']

            with torch.autocast('cuda', dtype=cfg.train.AMP_DTYPE, enabled=cfg.train.USE_AMP):
                outputs, _ = self.model(img, view_type_id, mask)
            outputs = outputs.float()      # loss in float32
            loss = self.criterion(outputs, targets) / self.accumulation_steps
            self.scaler.scale(loss).backward()
            running_loss += loss.detach() * img.size(0) * self.accumulation_steps

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)     # needed so clipping sees the real gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)         # was: xm.optimizer_step(...)
                self.scaler.update()
                self.optimizer.zero_grad()

        # flush any leftover partial accumulation at end of epoch
        if (i + 1) % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return to_scalar(running_loss) / (len(self.train_loader.dataset) + 1e-8)


    def valid_one_epoch(self, epoch):
        self.model.eval()

        running_loss = torch.tensor(0.0, device=self.device)
        tp = torch.tensor(0.0, device=self.device)
        fp = torch.tensor(0.0, device=self.device)
        fn = torch.tensor(0.0, device=self.device)

        raw_probs_chunks = []
        targets_chunks = []

        iterator = tqdm(self.valid_loader, desc=f"Epoch {epoch} [Valid]")

        with torch.no_grad():
            for batch in iterator:
                batch = move_batch(batch, self.device)
                img = batch['image']
                mask = batch['valid_mask']
                view_type_id = batch['view_type_ids']
                targets = batch['target']
                with torch.autocast('cuda', dtype=cfg.train.AMP_DTYPE, enabled=cfg.train.USE_AMP):
                    outputs, _ = self.model(img, view_type_id, mask)
                outputs = outputs.float()
                loss = self.criterion(outputs, targets)
                running_loss += loss.detach() * img.size(0)

                raw_probs = torch.sigmoid(outputs).float()
                # metrics reported here must match what submission will actually score,
                # so pF1/pF2 use corrected probs
                corrected_probs = torch.sigmoid(self.logit_adjustment(outputs)).float()

                tp += (targets * corrected_probs).sum()
                fp += ((1 - targets) * corrected_probs).sum()
                fn += (targets * (1 - corrected_probs)).sum()

                # AUC: rank-invariant to the shift, keep raw, gather once at epoch end
                raw_probs_chunks.append(raw_probs.detach())
                targets_chunks.append(targets.detach())

        # single GPU: no mesh_reduce needed, these are already the full totals
        running_loss = to_scalar(running_loss)
        tp = to_scalar(tp)
        fp = to_scalar(fp)
        fn = to_scalar(fn)

        epoch_loss = running_loss / (len(self.valid_loader.dataset) + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        epoch_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        epoch_f2 = 5 * tp / (5 * tp + 4 * fn + fp + 1e-8)   # pFbeta, beta=2, same tp/fp/fn

        # single GPU: no all_gather needed, just join the chunks
        global_probs = torch.cat(raw_probs_chunks).cpu().numpy().ravel()
        global_targets = torch.cat(targets_chunks).cpu().numpy().ravel()

        f2_best_trsh, best_f2_score = f2_best_threshold(global_targets, global_probs)
        f1_best_trsh, best_f1_score = f1_best_threshold(global_targets, global_probs)

        print(f"Global Optimal F1 Threshold: {f1_best_trsh:.4f} (Score: {best_f1_score:.4f})")
        print(f"Global Optimal F2 Threshold: {f2_best_trsh:.4f} (Score: {best_f2_score:.4f})")

        return epoch_loss, epoch_f1, epoch_f2


    def training(self):
        print(f"Starting Training on GPU...")
        patient_targets_full = self.train_slice.groupby('patient_id')['cancer'].max()
        neg_ids_full = patient_targets_full[patient_targets_full == 0].index.tolist()
        self.neg_sampler = BalancedPatientSampler(neg_ids_full, seed=self.neg_sampler_seed)
        if getattr(self, 'neg_sampler_state', None) is not None:
            self.neg_sampler.load_state_dict(self.neg_sampler_state)

        for epoch in range(self.start_epoch, self.epochs + 1):

            # -------------- random DownSampling every epoch
            self.p_real = self.train_slice['cancer'].mean()
            train_sub = create_balanced_patient_subset(self.train_slice, self.neg_sampler, cfg.glob.NEG_TO_POS_RATIO)
            self.p_train = train_sub['cancer'].mean()
            self.pos_weights = (train_sub['cancer'] == 0).sum() / (train_sub['cancer'] == 1).sum()
            pos_weight_tensor = torch.tensor([self.pos_weights], dtype=torch.float32).to(self.device)

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            train_ds = MammoDatasetVariousAugPerView(train_sub, transforms=self.train_transforms, split='train', random_crop_roi=True)
            val_ds   = MammoDatasetVariousAugPerView(self.val_slice, transforms=self.valid_transforms, split='train', random_crop_roi=False)
            print(f"\n Epoch {epoch} : p_real:{self.p_real:.4f} | p_train:{self.p_train:.4f} | pos_weights:{self.pos_weights:.4f} | length train_sub: {len(train_sub)} | length val: {len(self.val_slice)} | train_sub target count: {train_sub['cancer'].value_counts()} | batch_size = {cfg.train.BATCH_SIZE}")

            # --------------- Data Loaders (no DistributedSampler on one GPU)
            # generator seeded with epoch = same idea as sampler.set_epoch(epoch)
            g = torch.Generator()
            g.manual_seed(42 + epoch)

            self.train_loader = DataLoader(
                dataset=train_ds, batch_size=cfg.train.BATCH_SIZE, num_workers=cfg.train.NUM_WORKERS,
                shuffle=True, generator=g, drop_last=True, pin_memory=True,
            )
            self.valid_loader = DataLoader(
                dataset=val_ds, batch_size=cfg.train.BATCH_SIZE, num_workers=cfg.train.NUM_WORKERS,
                shuffle=False, drop_last=False, pin_memory=True,
            )
            # ----------------------------------------------------------
            tr_loss = self.train_one_epoch_accumulated(epoch)
            val_loss, val_f1_score, val_f2 = self.valid_one_epoch(epoch)

            # --- record history ---
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1_score)
            self.val_f2_scores.append(val_f2)

            print(f"\nEpoch {epoch} Summary: Train Loss = {tr_loss:.4f} | Valid Loss = {val_loss:.4f} | Valid Prob F1 = {val_f1_score:.4f} | Valid F2 = {val_f2:.4f} | LR = {self.optimizer.param_groups[0]['lr']:.6f}\n")

            self.scheduler.step()

            if epoch <= cfg.train.WARMUP_EPOCHS:
                # model isn't at full LR yet — don't let a noisy/weak warmup epoch
                # burn patience budget before real training has started
                if val_f1_score > self.best_f1:
                    self.best_f1 = val_f1_score
                    self._save_checkpoint(epoch, "best_model_checkpoint.pth")
                self.patience = 0
            else:
                if val_f1_score > self.best_f1:
                    self.best_f1 = val_f1_score
                    self._save_checkpoint(epoch, "best_model_checkpoint.pth")
                    print(f"Valid F1 improved to {val_f1_score:.4f}. Saved.\n")
                    self.patience = 0
                else:
                    self.patience += 1
            self._save_checkpoint(epoch, "last_checkpoint.pth")

            if self.patience >= cfg.train.PATIENCE:
                print("Early stopping triggered.")
                break

            del self.train_loader
            del self.valid_loader
            gc.collect()
            torch.cuda.empty_cache()     # NEW: free GPU memory between epochs
            time.sleep(2)

        plot_loss(
            self.val_losses,
            self.train_losses,
            self.val_f1_scores,
            self.val_f2_scores,
            "/kaggle/working/loss_f1_plot.png"
        )

