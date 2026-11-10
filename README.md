#  🎗️ Mammo: Multi-View Breast Cancer Detection

Deep learning pipeline for breast cancer detection from screening mammograms, built for the [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) challenge.

The model looks at **the two most popular views of each breast lateral** (e.g. CC and MLO) and predicts one cancer score per patient and side.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNetV2-green)
![Albumentations](https://img.shields.io/badge/Albumentations-Augmentation-purple)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Metrics-f7931e?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![DICOM](https://img.shields.io/badge/DICOM-Mammography-lightgrey)
![Kaggle](https://img.shields.io/badge/Kaggle-RSNA%20Breast%20Cancer-20beff?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

<table>
  <tr>
    <td align="center">
      <img src="./assets/images/output.png" width="400"><br>
      <sub>Distribution of several features</sub>
    </td>
    <td align="center">
      <img src="./assets/images/output1.png" width="400"><br>
      <sub>Age Distribution by Cnacer Status</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./assets/images/output2.png" width="400"><br>
      <sub>Correlation Heatmap of Feature</sub>
    </td>
    <td align="center">
      <img src="./assets/images/output3.png" height="300"><br>
      <sub>an image Sample</sub>
    </td>
  </tr>
  <!-- <tr>
    <td align="center">
      <img src="assets/images/SHAP.png" width="400"><br>
      <sub>CatBoost SHAP Analysis</sub>
    </td>
    <td align="center">
      <img src="assets/images/LIME.png" width="400"><br>
      <sub>XGBoost lime Analysis</sub>
    </td>
  </tr> -->
</table>

</div>

---

## ⚠️ Main Challenges

<!-- Add notes here -->
**Extreme Class Imbalance:** Positive cancer cases represent only a tiny fraction (about 2%) of the dataset, requiring careful handling to prevent the model from ignoring rare positive cases.

**High Tissue Density:**  Extremely dense breast tissue (Category D) appears white on X-rays and can easily hide malignant tumors (which also appear white), leading to challenging false negatives.

**Multi-View Dependencies:**  A single screening exam includes multiple image views per breast, meaning models must correctly aggregate predictions across images while avoiding data leakage between patients during cross-validation.

---
## 💡 key findings 

**Severe class imbalance**

Only 486 of 11,913 patients (~4%) have cancer. At image level it is ~2%.
Accuracy is misleading. Use class weights or sampling, and metrics like AUC and F1.

**Two views matter most**

CC and MLO make up almost all images (~27,000 each). ML, LM, AT and LMO are very rare.
Left and right breasts are balanced, so there is no side bias.

**Age is a weak signal**

Cancer cases are older (peak ~65–70) than non-cancer (~58).
The groups overlap a lot, so age alone cannot predict cancer.

**Density, implant, view and side show no special link**

Cancer is most common in density B and C, but this just matches the overall data.
Implants are rare (~1,500 images) and show no link to cancer. They may still be harder for the model because they look different.

**Data leakage risk**

Invasive (0.84), biopsy (0.61) and BIRADS are strongly linked to cancer, but they are only known after diagnosis.
so I do not use them as inputs.

**The model must learn from the images**

Age, side, view and implant have almost no correlation with cancer.
This is why the pipeline is image-based.

---
## 🧰 Pipeline

```mermaid
flowchart LR
    A[DICOM images] --> B[Normalize<br/>VOI LUT, invert MONOCHROME1, scale 0-255]
    B --> C[Breast ROI detection<br/>blur, threshold, largest contour]
    C --> D[Group by<br/>patient + laterality]
    D --> E[ROI crop<br/>1024 x 512]
    E --> F[Augmentation]
    F --> G[Model]
    G --> H[Cancer probability<br/>per breast]
```

- **Breast ROI cropping**: removes background and markers, so the model sees more tissue detail.
- **Multi-view input**: up to 2 views per breast, combined with a view-type embedding and max pooling.
- **Imbalance handling**: patient-level negative downsampling each epoch, `pos_weight` loss, and logit adjustment back to real prevalence.
- **Memory-friendly training**: mixed precision (fp16), gradient accumulation, and gradient checkpointing.
- **Metrics**: probabilistic F1 / F2 and best-threshold search.
- **Models** : EfficientNetV2-RW-M and Convnext_small
- **Image Size (before and after Pr Resizing)** : 1024 × 512 and 2048 × 1024

## 🤖 Model

```mermaid
flowchart TB
    V1[View 1<br/>1 x 1024 x 512] --> E1[Image Encoder<br/>shared weights]
    V2[View 2<br/>1 x 1024 x 512] --> E2[Image Encoder<br/>shared weights]
    E1 --> F1[Features]
    E2 --> F2[Features]
    T[View-type embedding] --> P
    F1 --> P[Add embedding + Max pooling<br/>masked for missing views]
    F2 --> P
    P --> H[MLP head<br/>LayerNorm - Linear - GELU - Dropout - Linear]
    H --> O[Logit]
```

## 🛠️ Training Configuration

| Category | Setting | Value |
|---|---|---|
| **Data** | Input size | 1024 × 512 → 2048 × 1024 ROI crop, 1 channel |
| | Views per breast | 2 (missing view is copied from the real one, and masked) |
| | ROI padding | 10 px |
| | Crop in training / validation | Random / center |
| | Validation split | 1 fold of 20 (patient-level, stratified) |
| **Progressive Resizing** | Resizing decision | After 12 epochs |
| | Initial input size | 1024 × 512 |
| | Initial LR | Backbone 1e-5, head 1e-4 |
| | Resized input size | 2048 × 1024 |
| | LR after resizing | Backbone 5e-6, head 5e-5 |
| **Imbalance** | Negative : positive ratio | 5 : 1 (patient-level, new negatives every epoch) |
| | Loss | BCE with `pos_weight` (negatives / positives) |
| | Logit adjustment | Shifts predictions back to real cancer rate |
| **Model** | Backbone | EfficientNetV2-RW-M and ConvNeXt-Small (both pretrained) |
| | View fusion | View-type embedding + max pooling |
| | Head | LayerNorm → Linear(256) → GELU → Dropout(0.3) → Linear(1) |
| **Optimization** | Optimizer | AdamW, weight decay 1e-4 |
| | Scheduler | 4 warm-up epochs + cosine decay (to 1e-8) |
| | Epochs | 50 |
| | Gradient clipping | 1.0 |
| **Efficiency** | Mixed precision (AMP) | fp16 with GradScaler |
| | Batch size / accumulation | 16 / 4 (effective batch 64) |
| | Gradient checkpointing | On |
| | BatchNorm | Frozen |
| **Augmentation** | Horizontal flip | p = 0.5 |
| | Shift / scale / rotate | 0.1 / 0.15 / 15°, p = 0.5 |
| | Brightness / contrast | ±0.1, p = 0.5 |
| | CLAHE | p = 0.4 |
| | Gamma / Gaussian noise | p = 0.2 each |
| **Evaluation** | Metrics | Probabilistic F1 and F2, best-threshold search |
| | Early stopping | Patience 8 (not counted during warm-up) |
| | Best model | Chosen by validation F1 |
| | Seed | 42 |

---

## 📁 Project Structure

```
.
├── configs.py                  # All settings (image size, batch size, LR, AMP...)
├── src/
│   ├── main.py                 # Entry point: split data, run training
│   ├── dataset/
│   │   ├── SaveROI.py          # Compute and save breast bounding boxes
│   │   ├── preprocessing.py    # Normalization, ROI detection, random crop
│   │   ├── augmentation.py     # Albumentations train / valid transforms
│   │   ├── MammoDatasetVariousAugPerView.py
│   │   └── BalancedPatientSampler.py
│   ├── models/
│   │   ├── ImageEncoder.py     # timm backbone (per view)
│   │   ├── ViewSetPooling.py   # max / mean / GeM pooling + head
│   │   ├── ViewSetAgg.py       # Transformer + attention MIL (alternative)
│   │   └── ModelHybrid.py      # Encoder + aggregator
│   ├── train/Trainer.py        # Training, validation, checkpointing
│   └── utils/utils.py          # Metrics, logit adjustment, helpers
└── assets/data/                # train.csv, train_with_bbox.csv
```

---

## ▶️ Getting Started

**1. Install**

```bash
pip install torch timm albumentations opencv-python pydicom dicomsdl \
            pylibjpeg pylibjpeg-libjpeg pandas scikit-learn matplotlib tqdm
```

**2. Prepare ROI boxes** (run once)

```bash
python src/dataset/SaveROI.py
```

**3. Train**

```bash
python src/main.py
```

> Data and weight paths currently point to Kaggle (`/kaggle/input/...`). Change them in `Trainer.py`, `ImageEncoder.py`, and the dataset files to run locally.

---

## 🎯 Results

<!-- Add validation metrics, comparison tables, and plots here -->

---

## 🔍 Explainability (XAI)

<!-- Add Grad-CAM / saliency examples here -->

---


## 🚀 Future Work

**Ensembling:** Average predictions from several models and folds.

**Cross-validation:** Train on all folds instead of one, so the model sees more data.

**Clinical data:** Add age and other metadata as extra model input.

**Threshold tuning:** Pick the best decision threshold on out-of-fold predictions.

---

## ℹ️ Acknowledgements

- Data: [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- Backbone: [timm](https://github.com/huggingface/pytorch-image-models)

## License

MIT
