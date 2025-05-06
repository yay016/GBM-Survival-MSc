# 🧠 Survival Prediction in Glioblastoma
This repository contains code, plots and notebooks from my MSc thesis project on AI-based prediction of glioblastoma progression and patient survival following radiotherapy. The work integrates image analysis, clinical data, and machine learning to model and analyse survival outcomes. The dataset is sourced from **The Cancer Imaging Archive (TCIA)** and requires approved access through their platform.

## 📁 Repository Structure
- `image_vis.py`  
  Script for visualizing image registration results. When run, it opens a graphical interface allowing selection of patient, modality, and image processing options.

- `clinical_analysis_ML_survival.ipynb`  
  Jupyter notebook for analyzing overall survival (OS) and running machine learning pipelines using clinical and imaging-derived features.

- `dataset_pipeline.py`  
  Utility functions for dataset construction and preprocessing. Used in `preprocess.ipynb` to prepare data for analysis and modeling.

- `models_classification.py`  
  Deep learning models for classification tasks (e.g., binary survival prediction).

- `models_regression.py`  
  Deep learning models for regression tasks (e.g., continious survival time prediction).

- `structure_volume_extractor.py`  
  Script to extract and analyze dose-volume and anatomical structure data from patient DICOM files.

- `dl-pfs-train.ipynb`  
  Notebook for deep learning-based progression-free survival (PFS) regression.

- `dl-survival.ipynb`  
  Notebook for deep learning-based survival prediction.

- `dose_distribution_analysis_survival.py`  
  Voxel-wise comparative analysis of 3D radiation dose distributions between short- and long-survival patient groups. Computes spatial dose statistics, significance maps, and summary metrics such as dose-falloff gradients and radii of gyration. Results are saved as plots.

- `GradCamPlots/`  
  Contains all Grad-CAM visualizations generated during model evaluation.

## 🔐 Data Access

This project uses sensitive clinical and imaging data from TCIA. To use the code:

1. Request access to the dataset through [TCIA](https://www.cancerimagingarchive.net/).
2. Download and organize the data according to the expected folder structure.

## 📂 Directory Structure

The code expects the dataset to follow the structure used in the **Burdenko Glioblastoma Progression** collection from TCIA. After downloading and extracting the data, it should be organized as follows:

```text
Burdenko-GBM-Progression/
├── Burdenko-GBM-001/
│   └── Radiotherapy planning/
│       ├── CT/
│       ├── MR T2FLAIR/
│       ├── MR CET1/
│       ├── RTSTRUCT/
│       ├── RTPLAN/
│       └── RTDOSE/
├── Burdenko-GBM-002/
│   └── Radiotherapy planning/
│       └── ...
└── ...
```
Each patient folder (e.g., `Burdenko-GBM-001`) must include a subdirectory named `Radiotherapy planning/`, containing separate series or folders for:
- **CT**
- **MR T2FLAIR**
- **MR CET1**
- **RTSTRUCT**
- **RTPLAN**
- **RTDOSE**

Set the `base_path` parameter in your scripts to the root dataset directory:

```python
base_path = "Burdenko-GBM-Progression"
```
## 💡 Example Usage
To process and load data for a specific patient:
```python
processor = DICOMProcessor(patient_id="Burdenko-GBM-001", base_path="Burdenko-GBM-Progression")
processor.load_all_data()
```
>📌**Note:** Due to privacy restrictions, the dataset is not included in this repository.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
