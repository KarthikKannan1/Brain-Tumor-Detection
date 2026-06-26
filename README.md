# Brain Tumor Detection: Wavelet + GLCM + Neural Network

A classical machine learning pipeline that detects and classifies brain tumors from MRI scans. The idea is simple: pull wavelet and texture features out of each MRI image, classify them with a small neural network, and then segment the likely tumor region.

> Research and educational project only. This is not a clinical or diagnostic tool.

## How it works

```
MRI image
   │
   ├─ 2D Discrete Wavelet Transform (DWT)  ->  approximation + detail subbands
   │
   ├─ Feature extraction per subband:
   │     GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
   │     Statistics (mean, variance, skewness, kurtosis, entropy)
   │     ->  44-dimensional feature vector
   │
   ├─ MLP classifier (shared backbone, swappable head)
   │     Binary head     -> tumor vs. no-tumor
   │     Multiclass head -> 4 classes
   │
   └─ Fuzzy C-Means clustering  ->  unsupervised tumor-region segmentation
```

The wavelet and GLCM stages are shared across both classification tasks. Only the output head and the labels change.

## Results

All numbers are on the held-out test set (1,600 images, 400 per class).

### Binary detection: 94.5% accuracy

| Class | Precision | Recall | F1 |
|---|---|---|---|
| no-tumor | 0.82 | 0.99 | 0.90 |
| tumor | 1.00 | 0.93 | 0.96 |

<img width="532" height="428" alt="image" src="https://github.com/user-attachments/assets/3557ba7c-fdc7-4a02-a5a2-274b13967acf" />

### 4-class classification: 84.0% accuracy

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.85 | 0.62 | 0.72 |
| meningioma | 0.75 | 0.79 | 0.77 |
| no-tumor | 0.85 | 0.99 | 0.92 |
| pituitary | 0.91 | 0.96 | 0.94 |

<img width="724" height="648" alt="image" src="https://github.com/user-attachments/assets/1e894227-eca8-4589-9743-bafb9aa2fa25" />

A few honest notes on these results. The model nails no-tumor and pituitary almost every time. Most of its mistakes come from confusing glioma with meningioma, which is actually a known hard case in brain tumor MRI work since the two can look quite similar on a scan. On the binary side, tumor recall is 0.93, so roughly 7% of tumors get missed. If this were ever a real clinical tool, that's the number you'd care about most, because a missed tumor is far worse than a false alarm.

## Dataset

I used the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which has 7,023 T1-weighted MRI images across the four classes. The Training split gets divided into train and validation using stratified sampling, and the Testing split is kept aside for the final evaluation.

The dataset isn't committed to this repo (it's gitignored). To run things yourself, grab it from Kaggle and drop it into a `data/` folder with `Training/` and `Testing/` subfolders, each holding the four class folders.

## Project structure

```
brain-tumor-detection/
├── src/
│   ├── data.py            # image loading, binary + multiclass labels, stratified split
│   ├── preprocessing.py   # DWT + GLCM/statistical feature extraction (cached)
│   ├── model.py           # MLP with swappable binary/multiclass head, training, evaluation
│   └── segmentation.py    # fuzzy c-means tumor-region segmentation
├── notebooks/
│   └── pipeline.ipynb     # end-to-end: load -> features -> train -> evaluate -> segment
├── requirements.txt
├── .gitignore
└── README.md
```

## Running it

```bash
# 1. Create an environment (Python 3.11 or 3.12 works well)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset (see above) into data/

# 4. Open the notebook and run it top to bottom
jupyter notebook notebooks/pipeline.ipynb
```

The first run extracts features for around 7,200 images, which takes a few minutes on the CPU. After that, it caches them, so every run afterward is quick.

## Background

This reimplements a paper I co-authored as an undergrad, *"Brain Tumor Detection by SWT-based Image Fusion with Neural Network."* The journal it was published in is no longer around, so I rebuilt the approach as a clean, open, reproducible project, this time with the kind of quantitative evaluation the original write-up was missing.

## License and use

Educational and research use only. Please don't use it for clinical decisions. If you build on it, a mention is appreciated.
