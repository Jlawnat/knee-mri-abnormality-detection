# Knee MRI Abnormality Detection
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-MRI-green)
![Self-Supervised Learning](https://img.shields.io/badge/Learning-Self--Supervised-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
Self-supervised learning for multi-label knee MRI abnormality detection using **2.5D DICOM inputs**, a **ResNet-18 encoder**, and supervised fine-tuning for **12 clinical targets**.

## Overview

The main challenge in this project is severe **label scarcity**.

The RSNA Knee Abnormality Detection training set contains **4,407 MRI studies**, but only **58 studies have complete labels** for the 12 target abnormalities.

Instead of training a deep learning model from scratch on only 58 labelled studies, this project uses the remaining **4,349 unlabelled MRI studies for self-supervised learning (SSL)**.

The overall strategy is:

```text
4,349 unlabelled MRI studies
            ↓
MRI preprocessing
            ↓
2.5D slice construction
            ↓
Self-supervised pretraining
            ↓
ResNet-18 MRI encoder
            ↓
58 labelled MRI studies
            ↓
Supervised fine-tuning
            ↓
12 abnormality probabilities
```

The goal is to learn useful MRI representations from the large unlabelled dataset before introducing the very limited supervised labels.

---

## Prediction Targets

The model predicts 12 binary knee abnormalities:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's cyst
11. Contusion
12. Fracture

---

## Dataset

### Original training data

| Dataset component      | Count |
| ---------------------- | ----: |
| Total training studies | 4,407 |
| Fully labelled studies |    58 |
| Unlabelled studies     | 4,349 |

Because fewer than 2% of the studies contain complete supervised labels, the unlabelled MRI data becomes the main source of representation learning.

---

## Self-Supervised Dataset

The SSL preprocessing pipeline produced:

| Component                       |   Count |
| ------------------------------- | ------: |
| Unlabelled studies              |   4,349 |
| MRI series                      |  24,035 |
| Indexed DICOM slices            | 808,550 |
| Original 2.5D triplets          | 760,480 |
| Corrupted DICOM slices detected |       2 |
| Final clean triplets            | 760,478 |
| SSL training studies            |   3,914 |
| SSL validation studies          |     435 |
| Training series                 |  21,640 |
| Validation series               |   2,395 |

To prevent studies with unusually large numbers of slices from dominating SSL training, the training pipeline samples **4 triplets per series per epoch**.

This results in:

* **86,560 SSL training samples per epoch**
* **9,580 fixed validation samples**

---

## MRI Preprocessing

MRI preprocessing is shared between self-supervised learning and supervised fine-tuning.

### 1. Physical slice ordering

DICOM files are ordered using physical MRI slice location rather than filename ordering.

Image orientation and position metadata are used to calculate each slice's physical position.

### 2. Intensity normalization

Each MRI series is independently normalized using its:

* 1st percentile intensity
* 99th percentile intensity

Values outside this range are clipped before scaling to approximately `[0, 1]`.

This helps reduce the effect of extreme intensity values and differences between MRI acquisitions.

### 3. Spacing-aware resizing

MRI images can have different pixel spacing.

Images are resized toward a common target spacing of:

```text
0.5 mm
```

before being cropped or padded.

### 4. Fixed image size

Each processed slice is transformed to:

```text
224 × 224
```

### 5. 2.5D representation

Instead of using a single MRI slice, three physically adjacent slices are combined:

```text
Channel 1 → previous slice
Channel 2 → centre slice
Channel 3 → next slice
```

The resulting model input is:

```text
3 × 224 × 224
```

This provides limited 3D anatomical context while retaining the efficiency of a 2D CNN.

---

## Corrupted DICOM Handling

Two corrupted DICOM slices were identified during preprocessing.

Rather than removing their entire MRI series, the pipeline:

1. identified unreadable slices,
2. recalculated intensity statistics using valid slices,
3. removed only 2.5D triplets touching the corrupted slices.

This reduced:

```text
760,480 triplets
        ↓
760,478 clean triplets
```

while preserving all **24,035 usable MRI series**.

---

## Self-Supervised Learning

The SSL stage uses a **ResNet-18 encoder** to learn MRI representations from the 4,349 unlabelled studies.

### Backbone

```text
ResNet-18
```

### Input

```text
3 × 224 × 224 2.5D MRI triplets
```

### Training strategy

* Study-level train/validation split
* Balanced sampling across MRI series
* MRI-safe image augmentations
* AdamW optimisation
* Learning-rate scheduling
* GPU mixed-precision training
* Best-model checkpointing

### SSL training

```text
Epochs: 5
Training samples per epoch: 86,560
Validation samples: 9,580
```

The best SSL model achieved:

```text
Best validation loss: 0.7054
```

The learned encoder is saved as:

```text
best_encoder.pt
```

and transferred to the supervised classification stage.

---

## Supervised Fine-Tuning

Only the **58 fully labelled studies** are used for supervised learning.

### Labelled dataset

| Component          | Count |
| ------------------ | ----: |
| Labelled studies   |    58 |
| MRI series         |   336 |
| 2.5D triplets      | 9,856 |
| Training studies   |    46 |
| Validation studies |    12 |

The split is performed at the **study level**.

This is essential because multiple MRI triplets belong to the same study. Splitting individual slices or triplets randomly would create data leakage between training and validation.

---

## Handling Class Imbalance

Several abnormalities are much rarer than others.

For example, in the 46-study training split:

| Target           | Positive | Negative |
| ---------------- | -------: | -------: |
| ACL              |       19 |       27 |
| MCL              |        7 |       39 |
| Medial Meniscus  |       21 |       25 |
| Lateral Meniscus |       19 |       27 |
| Medial OA        |       12 |       34 |
| Lateral OA       |        9 |       37 |
| PF OA            |       17 |       29 |
| Effusion         |       28 |       18 |
| Synovitis        |       21 |       25 |
| Baker's          |        9 |       37 |
| Contusion        |       15 |       31 |
| Fracture         |       14 |       32 |

Target-specific positive weights are therefore used with:

```python
BCEWithLogitsLoss
```

Rare abnormalities receive larger positive-class weights.

---

## Study-Balanced Sampling

MRI studies contain different numbers of slices and series.

If every triplet were sampled equally, studies containing more MRI slices could dominate training.

The supervised pipeline therefore uses inverse study-frequency sampling so that each study contributes more equally to model optimisation.

---

## Transfer Learning

The supervised model reuses the encoder learned during SSL.

```text
Pretrained ResNet-18 encoder
            ↓
512-dimensional MRI representation
            ↓
Dropout
            ↓
Linear classification layer
            ↓
12 logits
```

Different learning rates are used for the two model components:

```text
Encoder learning rate:     1e-5
Classifier learning rate:  1e-4
```

This allows the new classification layer to learn quickly while preserving useful MRI representations learned during SSL.

---

## Study-Level Prediction

Labels are defined at the **study level**, while the neural network operates on individual 2.5D triplets.

During validation and inference:

```text
MRI triplets
     ↓
Model logits
     ↓
Average logits across study
     ↓
Sigmoid
     ↓
12 study-level probabilities
```

This converts slice-level model outputs into one prediction per MRI study.

---

## Repository Structure

```text
knee-mri-abnormality-detection/
│
├── notebooks/
│   ├── 01_ssl_data_preparation.ipynb
│   ├── 02_ssl_pretraining.ipynb
│   └── 03_supervised_finetuning.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### `01_ssl_data_preparation.ipynb`

Builds the clean SSL dataset:

* MRI series inventory
* physical slice ordering
* 2.5D triplet generation
* study-level train/validation split
* intensity percentile calculation
* corrupted DICOM handling
* final preprocessing validation

### `02_ssl_pretraining.ipynb`

Implements the self-supervised learning pipeline:

* balanced MRI sampling
* SSL Dataset/DataLoader
* MRI augmentations
* ResNet-18 encoder
* GPU training
* validation
* checkpointing
* best encoder export

### `03_supervised_finetuning.ipynb`

Implements the supervised pipeline:

* 58-study labelled dataset
* labelled 2.5D triplets
* study-level train/validation split
* intensity preprocessing
* class weighting
* study-balanced sampling
* SSL encoder transfer
* supervised fine-tuning
* study-level evaluation
* test inference
* Kaggle submission generation

---

## Technology Stack

* Python
* PyTorch
* torchvision
* pydicom
* pandas
* NumPy
* scikit-learn
* Kaggle GPU environment

---

## Reproducibility

Random seeds are fixed where practical.

The project also uses **study-level splitting** throughout the pipeline to reduce the risk of leakage between related MRI slices.

The notebooks were developed and executed in Kaggle.

The dataset itself is **not included in this repository**.

To reproduce the project, obtain the RSNA Knee Abnormality Detection competition data through Kaggle and attach it to the notebooks.

---

## Installation

The main Python dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

## Results

### Self-Supervised Learning

```text
Best SSL validation loss: 0.7054
```

### Supervised Classification

Final supervised validation metrics and Kaggle submission results will be added after the completed inference run is verified.

Because only **58 studies contain complete labels**, single-split validation metrics should be interpreted carefully.

---

## Limitations

### Small labelled dataset

The main limitation is the extremely small supervised dataset.

Only:

```text
58 labelled studies
```

are available for fine-tuning.

This makes model evaluation highly sensitive to the particular train/validation split.

### 2.5D rather than full 3D modelling

The model uses neighbouring MRI slices as channels rather than processing complete 3D volumes.

This significantly reduces memory and computational requirements, but may lose some volumetric information.

### Study aggregation

The current approach averages predictions across all triplets belonging to a study.

More sophisticated attention-based or plane-aware aggregation could potentially improve performance.

---

## Future Improvements

Potential extensions include:

* multi-fold cross-validation
* model ensembling
* plane-specific MRI encoders
* attention-based study aggregation
* separate axial, sagittal, and coronal feature extraction
* stronger self-supervised objectives
* 3D CNN or transformer architectures
* test-time augmentation
* probability calibration
* pseudo-labelling of unlabelled studies

With such a small labelled dataset, **cross-validation and model averaging** are likely to be particularly valuable improvements.

---

## Key Idea

The core idea behind the project is simple:

> **When labels are scarce but unlabelled medical images are abundant, learn the representation first and the classification task second.**

Rather than discarding thousands of unlabelled MRI studies, the project uses them to learn reusable anatomical representations before fine-tuning on the limited labelled dataset.

---

## License

The code in this repository is released under the MIT License.

The RSNA competition dataset remains subject to its original data-use and competition terms and is not redistributed in this repository.
