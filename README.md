# MSc Thesis – Survival Prediction in Glioblastoma

This repository contains code and notebooks used in my MSc thesis project, focusing on survival prediction in glioblastoma patients through image analysis, clinical data, and machine learning. The dataset is sourced from **The Cancer Imaging Archive (TCIA)** and requires access approval through their platform.

## 📁 Repository Structure

- `3d_vis.py`  
  Script for visualizing image registration results. When run, it opens a graphical interface allowing selection of patient, modality, and image processing options.

- `clinical_analysis_ML_survival.ipynb`  
  Jupyter notebook for analyzing overall survival (OS) and running machine learning pipelines using clinical and imaging-derived features.

- `dataset_pipeline.py`  
  Utility functions for dataset construction and preprocessing. Used in `preprocess.ipynb` to prepare data for analysis and modeling.

- `models_classification.py`  
  Deep learning models for classification tasks (e.g., binary survival prediction).

- `models_regression.py`  
  Deep learning models for regression tasks (e.g., survival time prediction).

- `structure_volume_extractor.py`  
  Script to extract and analyze dose-volume and anatomical structure data from patient DICOM files.

- `dl-pfs-train.ipynb`  
  Notebook for deep learning-based progression-free survival (PFS) regression.

- `dl-survival.ipynb`  
  Notebook for deep learning-based survival prediction.

- `dose_distribution_analysis_survival.py`  
  Voxel-wise comparative analysis of 3D radiation dose distributions between short- and long-survival patient groups. Computes spatial dose statistics, significance maps, and summary metrics such as dose-falloff gradients and radii of gyration. Results are saved as plots.

- `GradCAM/`  
  Contains all Grad-CAM visualizations generated during model evaluation.

## 📦 Data Access

This project uses sensitive clinical and imaging data from TCIA. To use the code:

1. Request access to the dataset through [TCIA](https://www.cancerimagingarchive.net/).
2. Download and organize the data according to the expected folder structure.

>  **Note:** Due to privacy restrictions, the dataset is not included in this repository.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
