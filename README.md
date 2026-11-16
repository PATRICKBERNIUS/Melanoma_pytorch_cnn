# Melanoma vs. Benign Skin Lesion Classification with a Custom PyTorch CNN

A convolutional neural network, built from scratch in PyTorch, that classifies color images of skin lesions as **benign** or **malignant (melanoma)**. The project started from a YouTube tutorial by **Mike Saint-Antoine** that trained a CNN on grayscale versions of these images. I extended it to color images and then worked on making the model as good as possible: a tunable architecture, hyperparameter search with Optuna, data augmentation, regularization, early stopping, and class weighting to reduce missed melanomas. 

> **Note:** This README was written with the help of AI (Claude). This is a personal learning project and is **not** a diagnostic tool.

---

## Results

Evaluated once on the held-out test set (1,000 images: 500 benign, 500 malignant).

| Model | Test AUC | Accuracy | Malignant recall | False-positive rate | Missed melanomas (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| **Final CNN** (class weight 3 on malignant) | **0.977** | **0.920** | **0.914** | 0.074 | **43** | 37 |
| Earlier unweighted CNN | 0.973 | 0.918 | 0.878 | 0.042 | 61 | 21 |

**Takeaway:** the final model misses about 30% fewer melanomas than the earlier model (61 → 43), in exchange for more false alarms (21 → 37). Overall ranking ability (AUC) stays about the same. For a screening setting, a missed melanoma is the more costly mistake, so this trade-off was intentional.

---

## Dataset

*Melanoma Skin Cancer Dataset of 10,000 Images* on Kaggle: [add dataset link here].

- **Training:** the benign class was trimmed to match the malignant count, giving a balanced set of 9,210 images, then split 80/20 into **7,368 training** and **1,842 validation** images.
- **Test:** the dataset's own test split, 500 benign and 500 malignant.
- Images are loaded in color with OpenCV and resized (100×100 for tuning, 200×200 for the final model).

The images are not included in this repository. Download them from Kaggle (see [How to run](#how-to-run)).

---

## Approach

### Model

`TunableNet` (in `tunable_net.py`) builds a CNN from a list of channel sizes, so depth and width can be tuned. Each block is:

`Conv2d → BatchNorm2d → ReLU → MaxPool2d (first N blocks only) → Dropout2d`

followed by `AdaptiveAvgPool2d → Linear → ReLU → Dropout → Linear (2 outputs)`.

**Final architecture** (about 2.1M parameters):

| Setting | Value |
|---|---|
| Conv layers | 5 (channels 48, 96, 192, 192, 192), kernel 3 |
| Max-pool layers | 3 |
| Adaptive average pool | 10×10 |
| Fully connected hidden units | 64 |
| Dropout2d / Dropout | 0.01 / 0.2 |

### Training

| Setting | Value |
|---|---|
| Image size | 200×200 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.005 |
| Batch size | 64 |
| Loss | Cross-entropy with class weights `[1.0, 3.0]` (benign, malignant) |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5) on validation loss |
| Early stopping | Patience 20 on validation loss; best weights restored |
| Augmentation (training only) | Horizontal and vertical flips (p = 0.5), ColorJitter (brightness 0.065, contrast 0.13, saturation 0.07, hue 0.01), RandomResizedCrop (scale 0.95 to 1.0) |

Augmentation is applied per image on the GPU each epoch. Validation and test images are not augmented. Validation AUC is tracked every epoch.

### Hyperparameter tuning (Optuna)

Tuning was done in stages at 100×100 to keep each trial fast, maximizing validation AUC. Each stage used its own study and fixed the best values from the stage before.

| Stage | What was searched | Best validation AUC |
|---|---|---|
| Early passes | Learning rate, weight decay, dropout, then augmentation strength | |
| Pass 4 | Architecture: depth, width, pooling, FC size (Adam) | 0.9833 |
| Pass 5 | Wider architecture search with AdamW, including channel growth and number of pools | 0.9847 |
| Pass 6 | Regularization and augmentation: dropout, label smoothing, flips, color jitter, rotation, cropping, LR schedule | 0.9859 |

`tuning_full_search.py` shows all of these search spaces combined into a single objective for reference. **It was not run;** the results above come from the staged searches in `tuning.ipynb`.


## Limitations

- Single dataset and single source. Performance on images from other cameras, clinics or skin tones is unknown.
- The test set was used to compare several models during development, so the reported test numbers are likely slightly optimistic.
- Validation AUC (about 0.984) was consistently higher than test AUC (about 0.977), so the test split appears somewhat harder than the validation split.
- Hyperparameters were tuned at 100×100, and the final model was trained at 200×200.
- Not clinically validated. This is a learning project, **not** a medical device or diagnostic tool.

---

## Repository structure

```
├── README.md
├── requirements.txt
├── process_data.ipynb        # load Kaggle images, resize, balance, shuffle, save .npy files
├── tunable_net.py            # TunableNet model definition
├── train_model_crc.ipynb     # final training run + single test-set evaluation
├── tuning.ipynb              # Optuna studies (staged hyperparameter search)
├── tuning_full_search.py     # combined search space, for reference (not run)
└── tutorial/                 # original files from the tutorial-based starting point
    ├── net_class.py
    ├── train_model.ipynb
    └── test_model.ipynb
```

---

## How to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the data** from the Kaggle dataset above and unzip it so the folders are:
   ```
   melanoma_cancer_dataset/train/benign/
   melanoma_cancer_dataset/train/malignant/
   melanoma_cancer_dataset/test/benign/
   melanoma_cancer_dataset/test/malignant/
   ```
3. **Preprocess:** in `process_data.ipynb`, set `img_size` (100 or 200) and run all cells. This writes `melanoma_training_data_{img_size}.npy` and `melanoma_testing_data_{img_size}.npy`.
4. **Train and test:** run `train_model_crc.ipynb`. It trains with early stopping, saves `saved_model.pth`, plots the loss curves, and reports test accuracy, AUC, malignant recall, false-positive rate and the confusion matrix.
5. **(Optional) Tune:** run `tuning.ipynb`. Results are stored in a local SQLite file (`study.db`), so an interrupted study can be resumed.

A CUDA GPU is strongly recommended for training and tuning.

---

## Acknowledgements

- This project began as a follow-along of **Mike Saint-Antoine's** YouTube tutorial on building a CNN in PyTorch to classify melanoma images. The original tutorial used grayscale images; everything beyond that starting point (color pipeline, tunable architecture, Optuna tuning, augmentation, regularization and class weighting) was added afterward.
- Dataset: *Melanoma Skin Cancer Dataset of 10,000 Images* on Kaggle.
- This README was written with the help of AI (Claude).
"# Melanoma_pytorch_cnn" 
