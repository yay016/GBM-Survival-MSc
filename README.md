This repository contains code and notebooks used in my MSc thesis project, focusing on survival prediction in glioblastoma patients through image analysis, clinical data, and machine learning. The dataset used is from The Cancer Imaging Archive (TCIA), and access must be requested through their platform.

All GradCAM-plots are in the GradCAM folder. 

Repository Structure

3d_vis.py
A script for visualizing image registration results. When run, it opens a graphical interface where the user can select a patient, imaging modality, and image processing options.

clinical_analysis_ML_survival.ipynb
Jupyter notebook containing overall survival (OS) analysis and a machine learning pipeline for survival prediction based on clinical and imaging-derived features.

dataset_pipeline.py
Utility functions for dataset construction and preprocessing. This script is used in the notebook preprocess.ipynb to prepare data for analysis and modeling.

models_classification.py
Python script containing deep learning models for classification tasks (e.g., binary survival prediction).

models_regression.py
Python script containing deep learning models for regression tasks (e.g., survival time prediction).

structure_volume_extractor.py
Utility script used to extract and analyze dose-volume and anatomical structure data from the patient DICOM files.

dl-pfs-train.ipynb
Notebook for deep learning PFS regression.

dl-survival.ipynb
Notebook for deep learning survival prediction. 

dose_distribution_analysis_survival.py
This code performs a voxel-wise comparative analysis of 3D radiation dose distributions between two patient groups (short vs. long survival after glioblastoma treatment), computing spatial dose statistics, significance maps, and summary metrics such as dose-falloff gradients and radii of gyration, and visualizing the results through plots saved to disk.

Data Access

This project uses sensitive medical imaging and clinical data available through TCIA. To use the code, you must:

Request access to the dataset through TCIA.
Download and organize the data according to the project’s expected structure.
Note: Due to privacy restrictions, the dataset itself is not included in this repository.

Users can install the dependencies by running:
pip install -r requirements.txt
