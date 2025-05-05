"""
This script implements a pipeline for processing multimodal DICOM data (CT, MR, and RT images). 
It performs image registration, bias correction, resampling and croppin, ROI extraction, dose extraction, 
and deep learning dataset preparation.
"""
import cv2
import os
import re
import logging
from collections import OrderedDict
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# Suppress SimpleITK warnings globally
sitk.ProcessObject.SetGlobalWarningDisplay(False)


def resample_image(volume: np.ndarray, original_spacing: tuple, target_spacing: tuple, target_shape: tuple) -> np.ndarray:
    """
    Resamples a 3D volume to the target spacing and shape using linear interpolation.

    Parameters:
      volume (np.ndarray): Input volume with shape (depth, height, width).
      original_spacing (tuple): Original voxel spacing (z, y, x).
      target_spacing (tuple): Desired voxel spacing (z, y, x).
      target_shape (tuple): Desired output shape (depth, height, width).

    Returns:
      np.ndarray: Resampled volume with shape matching target_shape.
    """
    return resize(volume, target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)


def bias_correction(mr_image_sitk):
    """
    Applies N4 Bias Field Correction to an MR image using downsampling for faster processing.

    Parameters:
      mr_image_sitk (SimpleITK.Image): The input MR volume.

    Returns:
      SimpleITK.Image: Bias-corrected MR image at original resolution.
    """
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([100, 100, 100])
    corrector.SetConvergenceThreshold(0.0001)
    shrink_factor = 2
    mr_image_sitk = sitk.Cast(mr_image_sitk, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(mr_image_sitk, 0, 1, 200)
    maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
    mr_image_downsampled = sitk.Shrink(mr_image_sitk, [shrink_factor] * mr_image_sitk.GetDimension())
    maskImage_downsampled = sitk.Shrink(maskImage, [shrink_factor] * maskImage.GetDimension())
    _ = corrector.Execute(mr_image_downsampled, maskImage_downsampled)
    log_bias_field_downsampled = corrector.GetLogBiasFieldAsImage(mr_image_downsampled)
    log_bias_field_upsampled = sitk.Resample(
        log_bias_field_downsampled,
        mr_image_sitk,
        sitk.Transform(),
        sitk.sitkBSpline,
        0.0,
        log_bias_field_downsampled.GetPixelID()
    )
    mr_image_corrected = mr_image_sitk / sitk.Exp(log_bias_field_upsampled)
    return mr_image_corrected


def format_patient_id(patient_id):
    """
    Formats a patient ID to the 'Burdenko-GBM-xxx' format.

    Parameters:
      patient_id (str or int): The patient identifier.

    Returns:
      str: The formatted patient ID.
    """
    if isinstance(patient_id, str) and 'Burdenko-GBM-' in patient_id:
        return patient_id
    elif isinstance(patient_id, int) or (isinstance(patient_id, str) and patient_id.isdigit()):
        return f"Burdenko-GBM-{int(patient_id):03d}"
    else:
        raise ValueError(f"Invalid patient ID: {patient_id}")


class DICOMProcessor:
    """
    Processes DICOM data for a given patient, handling CT, MR, RTSTRUCT, RTPLAN, and RTDOSE data.
    """
    def __init__(self, patient_id, base_path="Burdenko-GBM-Progression", verbose=True,
                 standard_spacing=(1.0, 1.0, 1.0), bias_correction_enabled=True, apply_cropping=True):
        self.verbose = verbose
        self.patient_id = patient_id
        self.base_path = base_path
        self.patient_path = os.path.join(self.base_path, self.patient_id)
        self.standard_spacing = standard_spacing
        self.bias_correction_enabled = bias_correction_enabled
        self.apply_cropping = apply_cropping

        # Paths for various series and files
        self.ct_series_path = None
        self.mr_t2_series_path = None
        self.mr_t1_series_path = None
        self.rtstruct_files = []
        self.rtplan_files = []
        self.rtdose_files = []
        self.rtdose_series_paths = []

        # Image objects (SimpleITK.Images) and numpy arrays
        self.ct_image_sitk = None
        self.mr_image_sitk = None
        self.mr_image_original_sitk = None
        self.mr_t1_image_sitk = None
        self.mr_image_original_t1_sitk = None
        self.registered_mr_sitk = None
        self.registered_mr_t1_sitk = None
        self.final_transform = None
        self.final_transform_t1 = None
        self.dose_image_sitk = None

        self.ct_array = None
        self.mr_array = None
        self.mr_t1_array = None
        self.registered_mr_array = None
        self.registered_mr_t1_array = None
        self.dose_array = None

        # ROI and dose/beam related data
        self.slice_contours = {}
        self.all_structure_names = set()
        self.structure_masks = {}
        self.beam_geometries = []
        self.ct_slice_z_positions = []
        self.number_of_fractions_planned = -1
        self.delivery_maximum_dose = -1.0
        self.prescription_dose = -1.0
        self.origin_shift = np.zeros(3)
        self.original_spacing = None

        self.load_metadata()

    def load_metadata(self):
        if not os.path.isdir(self.patient_path):
            raise FileNotFoundError(f"Patient path not found: {self.patient_path}")
        studies = [d for d in os.listdir(self.patient_path)
                   if os.path.isdir(os.path.join(self.patient_path, d))]
        rt_study = next((s for s in studies if "Radiotherapy planning" in s), None)
        if not rt_study:
            raise FileNotFoundError("No radiotherapy study found for patient path.")
        rt_study_path = os.path.join(self.patient_path, rt_study)
        series = [d for d in os.listdir(rt_study_path)
                  if os.path.isdir(os.path.join(rt_study_path, d))]
        for s in series:
            if s.startswith('._'):
                continue
            series_path = os.path.join(rt_study_path, s)
            dicom_files = self.find_dicom_files(series_path)
            if not dicom_files:
                continue
            first_dicom = dicom_files[0][1]
            modality = getattr(first_dicom, 'Modality', '')
            series_description = getattr(first_dicom, 'SeriesDescription', '')
            if modality == "CT":
                self.ct_series_path = series_path
            elif modality == "MR" and "T2FLAIR" in series_description:
                self.mr_t2_series_path = series_path
            elif modality == "MR" and "CET1" in series_description:
                self.mr_t1_series_path = series_path
            elif modality == "RTSTRUCT":
                self.rtstruct_files.extend(dicom_files)
            elif modality == "RTPLAN":
                self.rtplan_files.extend(dicom_files)
            elif modality == "RTDOSE":
                if len(dicom_files) == 1:
                    self.rtdose_files.append(dicom_files[0][0])
                else:
                    self.rtdose_series_paths.append(series_path)
        if self.ct_series_path:
            self.get_ct_slice_positions()

    def get_ct_slice_positions(self):
        ct_files = self.find_dicom_files(self.ct_series_path)
        if not ct_files:
            return
        ct_files.sort(key=lambda x: x[1].ImagePositionPatient[2])
        slice_positions = [(idx, data.ImagePositionPatient[2])
                           for idx, (path, data) in enumerate(ct_files)]
        self.slice_indices, self.z_positions = zip(*slice_positions)
        self.slice_indices = np.array(self.slice_indices)
        self.z_positions = np.array(self.z_positions)

    def adjust_origin_after_cropping(self, original_image, cropped_image, index):
        new_origin = original_image.TransformIndexToPhysicalPoint(index)
        cropped_image.SetOrigin(new_origin)
        return cropped_image

    def resample_image_itk(self, image_sitk, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
        original_spacing = image_sitk.GetSpacing()
        original_size = image_sitk.GetSize()
        new_size = [int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
                    for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(image_sitk.GetPixelIDValue())
        return resampler.Execute(image_sitk)

    def apply_cropping_to_images(self):
        # Ensure required images are loaded
        assert self.ct_image_sitk is not None, "CT image not loaded."
        assert self.registered_mr_sitk is not None, "Registered MR T2 image not loaded."
        assert self.registered_mr_t1_sitk is not None, "Registered MR T1 image not loaded."
        assert self.dose_image_sitk is not None, "Dose image not loaded."

        ct_image = self.ct_image_sitk
        mr_t2_image = self.registered_mr_sitk
        mr_t1_image = self.registered_mr_t1_sitk
        dose_image = self.dose_image_sitk

        # Create brain mask via Otsu thresholding and close gaps
        brain_mask = sitk.OtsuThreshold(mr_t2_image, 0, 1, 200)
        brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
        brain_mask = sitk.BinaryMorphologicalClosing(brain_mask, (2, 2, 2))

        # Find the largest connected component
        cc_filter = sitk.ConnectedComponentImageFilter()
        brain_cc = cc_filter.Execute(brain_mask)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(brain_cc)
        largest_label = max(label_stats.GetLabels(), key=lambda lbl: label_stats.GetPhysicalSize(lbl), default=None)
        if largest_label is None:
            raise ValueError("No connected component found in brain mask.")
        brain_mask = sitk.BinaryThreshold(brain_cc, lowerThreshold=largest_label, upperThreshold=largest_label,
                                            insideValue=1, outsideValue=0)

        # Compute bounding box with a margin
        bounding_box = label_stats.GetBoundingBox(largest_label)  # [x, y, z, size_x, size_y, size_z]
        index = list(bounding_box[:3])
        size = list(bounding_box[3:])
        margin_mm = 2
        spacing = mr_t2_image.GetSpacing()
        margin_voxels = [int(margin_mm / sp) for sp in spacing]
        index = [max(0, idx - margin_voxels[i]) for i, idx in enumerate(index)]
        size = [min(mr_t2_image.GetSize()[i] - index[i], size[i] + 2 * margin_voxels[i]) for i in range(3)]

        # Adjust to obtain a square bounding box in x and y
        target_size_xy = max(size[0], size[1])
        if size[0] < target_size_xy:
            delta = target_size_xy - size[0]
            index[0] = max(0, index[0] - delta // 2)
            size[0] = target_size_xy
        if size[1] < target_size_xy:
            delta = target_size_xy - size[1]
            index[1] = max(0, index[1] - delta // 2)
            size[1] = target_size_xy
        size[0] = min(size[0], mr_t2_image.GetSize()[0] - index[0])
        size[1] = min(size[1], mr_t2_image.GetSize()[1] - index[1])

        if self.verbose:
            print(f"Cropping parameters: index={index}, size={size}")

        try:
            cropped_ct = sitk.RegionOfInterest(ct_image, size=size, index=index)
            cropped_mr_t2 = sitk.RegionOfInterest(mr_t2_image, size=size, index=index)
            cropped_mr_t1 = sitk.RegionOfInterest(mr_t1_image, size=size, index=index)
            cropped_dose = sitk.RegionOfInterest(dose_image, size=size, index=index)
        except Exception as e:
            raise ValueError(f"Error during cropping: {e}")

        # Adjust origins after cropping and save the origin shift
        self.ct_image_sitk = self.adjust_origin_after_cropping(ct_image, cropped_ct, index)
        self.registered_mr_sitk = self.adjust_origin_after_cropping(mr_t2_image, cropped_mr_t2, index)
        self.registered_mr_t1_sitk = self.adjust_origin_after_cropping(mr_t1_image, cropped_mr_t1, index)
        self.dose_image_sitk = self.adjust_origin_after_cropping(dose_image, cropped_dose, index)
        self.origin_shift = np.array(ct_image.GetOrigin()) - np.array(self.ct_image_sitk.GetOrigin())

    def load_all_data(self):
        # Load CT series
        if self.ct_series_path and self.ct_image_sitk is None:
            self.ct_image_sitk = self.load_dicom_series(self.ct_series_path)
            if self.ct_image_sitk is not None:
                self.ct_array = sitk.GetArrayFromImage(self.ct_image_sitk)
                self.original_spacing = tuple(self.ct_image_sitk.GetSpacing())
                self.volume_shape = self.ct_array.shape
            else:
                if self.verbose:
                    print(f"CT image could not be loaded for patient {self.patient_id}")

        # Load MR T2 series and apply bias correction if enabled
        if self.mr_t2_series_path and self.mr_image_sitk is None:
            self.mr_image_sitk = self.load_dicom_series(self.mr_t2_series_path)
            self.mr_image_original_sitk = self.mr_image_sitk
        if self.mr_image_sitk is not None:
            if self.bias_correction_enabled:
                self.mr_image_sitk = bias_correction(self.mr_image_sitk)
            self.mr_array = sitk.GetArrayFromImage(self.mr_image_sitk)
        else:
            if self.verbose:
                print(f"MR T2 image could not be loaded for patient {self.patient_id}")

        # Load MR T1 series and apply bias correction if enabled
        if self.mr_t1_series_path and self.mr_t1_image_sitk is None:
            self.mr_t1_image_sitk = self.load_dicom_series(self.mr_t1_series_path)
            self.mr_image_original_t1_sitk = self.mr_t1_image_sitk
            if self.mr_t1_image_sitk is not None:
                if self.bias_correction_enabled:
                    self.mr_t1_image_sitk = bias_correction(self.mr_t1_image_sitk)
                self.mr_t1_array = sitk.GetArrayFromImage(self.mr_t1_image_sitk)
            else:
                if self.verbose:
                    print(f"MR T1 image could not be loaded for patient {self.patient_id}")

        # Register MR T2 to CT
        if self.ct_image_sitk and self.mr_image_sitk and self.registered_mr_sitk is None:
            self.registered_mr_sitk, self.final_transform = self.register_mr_to_ct(self.ct_image_sitk, self.mr_image_sitk)
            if self.registered_mr_sitk is not None:
                self.registered_mr_array = sitk.GetArrayFromImage(self.registered_mr_sitk)
            else:
                if self.verbose:
                    print(f"MR T2 image could not be registered to CT for patient {self.patient_id}")

        # Register MR T1 to CT
        if self.ct_image_sitk and self.mr_t1_image_sitk and self.registered_mr_t1_sitk is None:
            self.registered_mr_t1_sitk, self.final_transform_t1 = self.register_mr_to_ct(self.ct_image_sitk, self.mr_t1_image_sitk)
            if self.registered_mr_t1_sitk is not None:
                self.registered_mr_t1_array = sitk.GetArrayFromImage(self.registered_mr_t1_sitk)
            else:
                if self.verbose:
                    print(f"MR T1 image could not be registered to CT for patient {self.patient_id}")

        # Resample CT and MR images to standard spacing
        if self.registered_mr_sitk is not None and self.ct_image_sitk is not None:
            if self.verbose:
                print("Resampling CT image to standard spacing...")
            self.ct_image_sitk = self.resample_image_itk(self.ct_image_sitk, new_spacing=self.standard_spacing)
            self.ct_array = sitk.GetArrayFromImage(self.ct_image_sitk)
            if self.verbose:
                print("Resampling registered MR T2 image to standard spacing...")
            self.registered_mr_sitk = self.resample_image_itk(self.registered_mr_sitk, new_spacing=self.standard_spacing)
            self.registered_mr_array = sitk.GetArrayFromImage(self.registered_mr_sitk)
            if self.registered_mr_t1_sitk is not None:
                if self.verbose:
                    print("Resampling registered MR T1 image to standard spacing...")
                self.registered_mr_t1_sitk = self.resample_image_itk(self.registered_mr_t1_sitk, new_spacing=self.standard_spacing)
                self.registered_mr_t1_array = sitk.GetArrayFromImage(self.registered_mr_t1_sitk)

        if self.ct_image_sitk and self.ct_array is not None and not hasattr(self, 'z_positions'):
            self.get_ct_slice_positions()
        if (self.rtdose_files or self.rtdose_series_paths) and self.dose_image_sitk is None:
            self.load_rtdose()
        if self.dose_image_sitk is not None:
            if self.verbose:
                print("Resampling RTDOSE image to standard spacing...")
            self.dose_image_sitk = self.resample_image_itk(self.dose_image_sitk, new_spacing=self.standard_spacing)
            self.dose_array = sitk.GetArrayFromImage(self.dose_image_sitk)
        if self.apply_cropping:
            self.apply_cropping_to_images()
        else:
            if self.verbose:
                print("Skipping cropping as it is disabled.")

        self.ct_array = sitk.GetArrayFromImage(self.ct_image_sitk)
        self.registered_mr_array = sitk.GetArrayFromImage(self.registered_mr_sitk)
        if self.registered_mr_t1_sitk is not None:
            self.registered_mr_t1_array = sitk.GetArrayFromImage(self.registered_mr_t1_sitk)
        self.dose_array = sitk.GetArrayFromImage(self.dose_image_sitk)
        self.volume_shape = self.ct_array.shape
        self.ct_slice_z_positions = [(idx, self.ct_image_sitk.TransformIndexToPhysicalPoint((0, 0, idx))[2])
                                     for idx in range(self.ct_image_sitk.GetSize()[2])]
        self.slice_indices, self.z_positions = zip(*self.ct_slice_z_positions)
        self.slice_indices = np.array(self.slice_indices)
        self.z_positions = np.array(self.z_positions)
        if self.rtstruct_files and not self.slice_contours:
            self.load_rtstruct()
        if self.rtplan_files and not self.beam_geometries:
            self.load_rtplan()

    def find_dicom_files(self, series_path):
        try:
            files = os.scandir(series_path)
            dicom_files = []
            for entry in files:
                if entry.name.startswith('._') or not entry.name.endswith(".dcm") or not entry.is_file():
                    continue
                filepath = entry.path
                try:
                    dicom_data = pydicom.dcmread(filepath, stop_before_pixels=True)
                    dicom_files.append((filepath, dicom_data))
                except (pydicom.errors.InvalidDicomError, AttributeError):
                    continue
            return dicom_files
        except FileNotFoundError:
            return []

    def load_dicom_series(self, series_path):
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(series_path)
        if not series_IDs:
            return None
        series_file_names = reader.GetGDCMSeriesFileNames(series_path, series_IDs[0])
        reader.SetFileNames(series_file_names)
        try:
            return reader.Execute()
        except Exception:
            return None

    def register_mr_to_ct(self, ct_image_sitk, mr_image_sitk):
        ct_image = sitk.Cast(ct_image_sitk, sitk.sitkFloat32)
        mr_image = sitk.Cast(mr_image_sitk, sitk.sitkFloat32)
        initial_transform = sitk.CenteredTransformInitializer(ct_image, mr_image, sitk.Euler3DTransform(),
                                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                     minStep=1e-6,
                                                                     numberOfIterations=200,
                                                                     gradientMagnitudeTolerance=1e-6)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        try:
            final_transform = registration_method.Execute(ct_image, mr_image)
        except Exception:
            return None, None
        try:
            registered_mr = sitk.Resample(mr_image_sitk, ct_image_sitk, final_transform,
                                          sitk.sitkLinear, 0.0, mr_image_sitk.GetPixelID())
        except Exception:
            return None, None
        return registered_mr, final_transform

    def load_rtstruct(self):
        rtstruct_data_list = [data for (path, data) in self.rtstruct_files
                              if not os.path.basename(path).startswith('._')]
        if not rtstruct_data_list:
            return
        self.structure_masks = {}
        z_positions = self.z_positions
        slice_indices = self.slice_indices
        tolerance = 1e-3
        for rtstruct_data in rtstruct_data_list:
            if not hasattr(rtstruct_data, 'StructureSetROISequence'):
                continue
            roi_dict = {}
            for roi in rtstruct_data.StructureSetROISequence:
                if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                    roi_name = roi.ROIName.strip()
                    roi_dict[roi.ROINumber] = roi_name
                    self.all_structure_names.add(roi_name)
                    if roi_name not in self.structure_masks:
                        self.structure_masks[roi_name] = np.zeros(self.volume_shape, dtype=np.uint8)
            if hasattr(rtstruct_data, 'ROIContourSequence'):
                for roi_contour in rtstruct_data.ROIContourSequence:
                    roi_number = roi_contour.ReferencedROINumber
                    roi_name = roi_dict.get(roi_number, None)
                    if roi_name is None:
                        continue
                    if hasattr(roi_contour, 'ContourSequence'):
                        for contour in roi_contour.ContourSequence:
                            contour_data = getattr(contour, 'ContourData', [])
                            if not contour_data:
                                continue
                            try:
                                contour_points = np.array(contour_data).reshape(-1, 3)
                            except ValueError:
                                continue
                            mean_z = np.mean(contour_points[:, 2])
                            distance = np.abs(z_positions - mean_z)
                            min_idx = np.argmin(distance)
                            if distance[min_idx] < tolerance:
                                idx = slice_indices[min_idx]
                                if idx not in self.slice_contours:
                                    self.slice_contours[idx] = []
                                pixel_coords = self.transform_physical_to_pixel(contour_points)
                                if pixel_coords is not None:
                                    self.slice_contours[idx].append({
                                        'points': pixel_coords,
                                        'structure_name': roi_name
                                    })
                                    if pixel_coords.shape[0] >= 3:
                                        poly = pixel_coords[:, :2]
                                        rr, cc = polygon(poly[:, 0], poly[:, 1], shape=self.volume_shape[1:])
                                        self.structure_masks[roi_name][idx, rr, cc] = 1
                            else:
                                continue
            else:
                continue

    def transform_physical_to_pixel(self, contour_points):
        try:
            origin = np.array(self.ct_image_sitk.GetOrigin())
            spacing = np.array(self.ct_image_sitk.GetSpacing())
            direction = np.array(self.ct_image_sitk.GetDirection()).reshape((3, 3))
            relative_points = contour_points - origin
            direction_inv = np.linalg.inv(direction)
            transformed_points = (direction_inv @ relative_points.T).T
            indices_cont = transformed_points / spacing
            indices = np.floor(indices_cont + 0.5).astype(int)
            pixel_coords = indices[:, [1, 0]]
            return pixel_coords
        except Exception:
            return None

    def load_rtplan(self):
        if not self.rtplan_files:
            return
        rtplan_dataset = self.rtplan_files[0][1]
        try:
            fraction_groups = rtplan_dataset.FractionGroupSequence
            for fg in fraction_groups:
                self.number_of_fractions_planned = getattr(fg, 'NumberOfFractionsPlanned', -1)
        except AttributeError:
            self.number_of_fractions_planned = -1
        try:
            dose_references = rtplan_dataset.DoseReferenceSequence
            for dose_ref in dose_references:
                self.delivery_maximum_dose = getattr(dose_ref, 'DeliveryMaximumDose', -1.0)
                self.prescription_dose = getattr(dose_ref, 'TargetPrescriptionDose', -1.0)
        except AttributeError:
            self.delivery_maximum_dose = -1.0
            self.prescription_dose = -1.0
        self.beam_geometries = self.get_beam_geometries(rtplan_dataset)

    def get_beam_geometries(self, rtplan_data):
        beam_geometries = []
        for beam in rtplan_data.BeamSequence:
            beam_info = {}
            beam_info['BeamName'] = getattr(beam, 'BeamName', f"Beam {beam.BeamNumber}")
            first_cp = beam.ControlPointSequence[0]
            beam_info['GantryAngle'] = first_cp.GantryAngle
            beam_info['CollimatorAngle'] = first_cp.BeamLimitingDeviceAngle
            beam_info['IsocenterPosition'] = first_cp.IsocenterPosition
            beam_info['SourceAxisDistance'] = beam.SourceAxisDistance
            beam_info['LeafPositions'] = []
            if hasattr(beam, 'BeamLimitingDeviceSequence'):
                for device in beam.BeamLimitingDeviceSequence:
                    if device.RTBeamLimitingDeviceType in ['MLCX', 'MLCY']:
                        leaf_positions = []
                        for cp in beam.ControlPointSequence:
                            if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                                for bldp in cp.BeamLimitingDevicePositionSequence:
                                    if bldp.RTBeamLimitingDeviceType == device.RTBeamLimitingDeviceType:
                                        leaf_positions.append(bldp.LeafJawPositions)
                        beam_info['LeafPositions'].append({
                            'DeviceType': device.RTBeamLimitingDeviceType,
                            'Positions': leaf_positions
                        })
            beam_geometries.append(beam_info)
        return beam_geometries

    def load_rtdose(self):
        dose_image_sitk, rtdose_dataset = self.load_rtdose_data()
        if dose_image_sitk is None:
            self.dose_image_sitk = sitk.Image(self.ct_image_sitk.GetSize(), sitk.sitkFloat32)
            self.dose_image_sitk.CopyInformation(self.ct_image_sitk)
            self.dose_array = sitk.GetArrayFromImage(self.dose_image_sitk)
            self.dose_array.fill(0)
        else:
            same_origin = np.allclose(dose_image_sitk.GetOrigin(), self.ct_image_sitk.GetOrigin())
            same_spacing = np.allclose(dose_image_sitk.GetSpacing(), self.ct_image_sitk.GetSpacing())
            same_direction = np.allclose(dose_image_sitk.GetDirection(), self.ct_image_sitk.GetDirection())
            if not (same_origin and same_spacing and same_direction):
                reg_dose, _ = self.register_dose_to_ct(self.ct_image_sitk, dose_image_sitk)
                if reg_dose is None:
                    self.dose_image_sitk = sitk.Image(self.ct_image_sitk.GetSize(), sitk.sitkFloat32)
                    self.dose_image_sitk.CopyInformation(self.ct_image_sitk)
                    self.dose_array = sitk.GetArrayFromImage(self.dose_image_sitk)
                    self.dose_array.fill(0)
                else:
                    self.dose_image_sitk = reg_dose
            else:
                self.dose_image_sitk = dose_image_sitk
            if self.dose_image_sitk is not None:
                self.dose_image_sitk = self.resample_image_itk(self.dose_image_sitk, new_spacing=self.standard_spacing)
                try:
                    self.dose_array = sitk.GetArrayFromImage(self.dose_image_sitk)
                except Exception:
                    self.dose_array = np.zeros(self.ct_image_sitk.GetSize()[::-1], dtype=np.float32)
            try:
                self.dose_grid_scaling = float(getattr(rtdose_dataset, 'DoseGridScaling', 1.0))
            except Exception:
                self.dose_grid_scaling = 1.0
            self.dose_array = self.dose_array * self.dose_grid_scaling
            if self.dose_array.shape != self.ct_array.shape:
                min_shape = np.minimum(self.dose_array.shape, self.ct_array.shape)
                self.dose_array = self.dose_array[:min_shape[0], :min_shape[1], :min_shape[2]]
                self.ct_array = self.ct_array[:min_shape[0], :min_shape[1], :min_shape[2]]
                if self.mr_array is not None:
                    self.mr_array = self.mr_array[:min_shape[0], :min_shape[1], :min_shape[2]]
                if self.mr_t1_array is not None:
                    self.mr_t1_array = self.mr_t1_array[:min_shape[0], :min_shape[1], :min_shape[2]]

    def load_rtdose_data(self):
        if self.rtdose_files:
            rtdose_file_path = self.rtdose_files[0]
            try:
                dose_image_sitk = sitk.ReadImage(rtdose_file_path)
                rtdose_dataset = pydicom.dcmread(rtdose_file_path, stop_before_pixels=True)
                return dose_image_sitk, rtdose_dataset
            except Exception:
                return None, None
        elif self.rtdose_series_paths:
            rtdose_series_path = self.rtdose_series_paths[0]
            dose_image_sitk = self.load_dicom_series(rtdose_series_path)
            if dose_image_sitk is None:
                return None, None
            rtdose_files_in_series = self.find_dicom_files(rtdose_series_path)
            if not rtdose_files_in_series:
                return dose_image_sitk, None
            rtdose_dataset = pydicom.dcmread(rtdose_files_in_series[0][0], stop_before_pixels=True)
            return dose_image_sitk, rtdose_dataset
        else:
            return None, None

    def register_dose_to_ct(self, ct_image_sitk, dose_image_sitk):
        ct_image = sitk.Cast(ct_image_sitk, sitk.sitkFloat32)
        dose_image = sitk.Cast(dose_image_sitk, sitk.sitkFloat32)
        initial_transform = sitk.CenteredTransformInitializer(ct_image, dose_image, sitk.Euler3DTransform(),
                                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                     minStep=1e-6,
                                                                     numberOfIterations=100,
                                                                     gradientMagnitudeTolerance=1e-6)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        try:
            final_transform = registration_method.Execute(ct_image, dose_image)
        except Exception:
            return None, None
        try:
            registered_dose = sitk.Resample(dose_image_sitk, ct_image_sitk, final_transform,
                                            sitk.sitkLinear, 0.0, dose_image_sitk.GetPixelID())
        except Exception:
            return None, None
        return registered_dose, final_transform

    def get_dose_array(self):
        if self.registered_dose is not None:
            try:
                return sitk.GetArrayFromImage(self.registered_dose)
            except Exception as e:
                self.logger.error(f"Error converting registered dose image to array: {e}")
                return None
        else:
            self.logger.error("No registered dose image available.")
            return None


class ProcessorCache:
    """
    Caches DICOMProcessor instances for patients to avoid reloading data repeatedly.
    """
    def __init__(self, max_size=10, base_path="Burdenko-GBM-Progression", verbose=False):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.base_path = base_path
        self.verbose = verbose

    def get(self, patient_id):
        formatted_id = format_patient_id(patient_id)
        if formatted_id in self.cache:
            self.cache.move_to_end(formatted_id)
            return self.cache[formatted_id]
        else:
            try:
                processor = DICOMProcessor(patient_id=formatted_id, base_path=self.base_path, verbose=self.verbose)
                processor.load_all_data()
                self.cache[formatted_id] = processor
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                return processor
            except Exception as e:
                if self.verbose:
                    print(f"Error loading DICOMProcessor for patient {formatted_id}: {e}")
                return None


def canonicalize_roi_name(raw_name):
    """
    Normalizes ROI names so that all variants of "brainstem" are captured.

    Examples:
      "BrainStem"    -> "Brainstem"
      "brain stem"   -> "Brainstem"
      "brain--stem"  -> "Brainstem"
      "Brain"        -> "Brain"
    """
    if not raw_name:
        return ""
    pattern = r'\bbrain[\s_-]*stem\b'
    if re.search(pattern, raw_name, re.IGNORECASE):
        return "Brainstem"
    return raw_name.strip()


class DoseLoader:
    """
    Loads and processes dose data for a patient. It reads RTDOSE, RTPLAN, and RTSTRUCT files,
    registers dose images to CT, and extracts dose and volume metrics.
    """
    def __init__(self, patient_id, base_path, dose_threshold_percentage=95.0, verbose=False):
        if isinstance(patient_id, int):
            patient_id = f"{patient_id:03d}"
        elif isinstance(patient_id, str) and patient_id.isdigit() and len(patient_id) < 3:
            patient_id = patient_id.zfill(3)
        self.patient_id = patient_id
        self.base_path = base_path
        self.dose_threshold_percentage = dose_threshold_percentage
        self.patient_path = os.path.join(base_path, patient_id)

        self.logger = logging.getLogger(f"DoseLoader_Patient_{self.patient_id}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.ct_image = None
        self.registered_doses = []
        self.registered_dose = None
        self.dose_grid_scaling = 1.0

        self.number_of_fractions_planned = -1
        self.delivery_maximum_dose = -1.0
        self.prescription_dose = -1.0

        self.rtstruct_masks = {}  # We only process ROIs that canonicalize to "Brainstem"

        self.process_patient()

    def find_dicom_files(self, series_path, modality_filter=None, description_filter=None):
        try:
            files = os.listdir(series_path)
        except Exception as e:
            self.logger.error(f"Error listing directory {series_path}: {e}")
            return []
        dicom_files = []
        for file in files:
            if file.startswith('._'):
                continue
            filepath = os.path.join(series_path, file)
            try:
                dicom_data = pydicom.dcmread(filepath, stop_before_pixels=True)
                if modality_filter and dicom_data.Modality != modality_filter:
                    continue
                if description_filter and description_filter not in getattr(dicom_data, 'SeriesDescription', ''):
                    continue
                dicom_files.append((filepath, dicom_data))
            except Exception as e:
                self.logger.warning(f"Skipping file {filepath}: {e}")
                continue
        return dicom_files

    def load_dicom_series(self, series_path):
        reader = sitk.ImageSeriesReader()
        try:
            series_IDs = reader.GetGDCMSeriesIDs(series_path)
            if not series_IDs:
                self.logger.error(f"No DICOM series found in {series_path}")
                return None
            dose_images = []
            for series_id in series_IDs:
                series_file_names = reader.GetGDCMSeriesFileNames(series_path, series_id)
                reader.SetFileNames(series_file_names)
                sitk_image = reader.Execute()
                try:
                    dicom_data = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
                    dose_grid_scaling = getattr(dicom_data, 'DoseGridScaling', 1.0)
                except Exception as e:
                    self.logger.warning(f"Could not read DoseGridScaling from {series_file_names[0]}: {e}")
                    dose_grid_scaling = 1.0
                sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32) * dose_grid_scaling
                dose_images.append(sitk_image)
                self.logger.info(f"Loaded DICOM series from {series_path} (Series ID: {series_id})")
            if dose_images:
                return max(dose_images, key=lambda img: sitk.GetArrayFromImage(img).max())
            return None
        except Exception as e:
            self.logger.error(f"Error loading DICOM series from {series_path}: {e}")
            return None

    def load_dicom_file(self, file_path):
        try:
            dicom_data = pydicom.dcmread(file_path)
            self.dose_grid_scaling = getattr(dicom_data, 'DoseGridScaling', 1.0)
            sitk_image = sitk.ReadImage(file_path)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32) * self.dose_grid_scaling
            self.logger.info(f"Loaded DICOM file {file_path}")
            return sitk_image
        except Exception as e:
            self.logger.error(f"Error loading DICOM file from {file_path}: {e}")
            return None

    def process_rtplan(self, rtplan_files):
        for rtplan_file in rtplan_files:
            self.logger.info(f"Processing RTPLAN file: {rtplan_file}")
            prescription_dose = self.extract_prescription_dose(rtplan_file)
            if prescription_dose > 0:
                self.prescription_dose = prescription_dose
                self.logger.info(f"Set prescription_dose to {self.prescription_dose} Gy")
                break
        else:
            self.logger.warning("No valid RTPLAN file found for prescription dose.")

    def extract_prescription_dose(self, rtplan_file):
        try:
            rtplan_data = pydicom.dcmread(rtplan_file)
            prescription_dose = -1.0
            if hasattr(rtplan_data, 'DoseReferenceSequence'):
                for dose_ref in rtplan_data.DoseReferenceSequence:
                    if hasattr(dose_ref, 'TargetPrescriptionDose'):
                        prescription_dose = float(dose_ref.TargetPrescriptionDose)
                        self.logger.info(f"Extracted prescription_dose {prescription_dose} Gy from {rtplan_file}")
                        break
            if hasattr(rtplan_data, 'FractionGroupSequence'):
                for fg in rtplan_data.FractionGroupSequence:
                    if hasattr(fg, 'NumberOfFractionsPlanned'):
                        self.number_of_fractions_planned = int(fg.NumberOfFractionsPlanned)
                        self.logger.info(f"Extracted NumberOfFractionsPlanned {self.number_of_fractions_planned} from {rtplan_file}")
                        break
            return prescription_dose
        except Exception as e:
            self.logger.error(f"Error extracting prescription dose from {rtplan_file}: {e}")
            return -1.0

    def register_images(self, fixed_image, moving_image):
        try:
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
            moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                         minStep=1e-6,
                                                                         numberOfIterations=200,
                                                                         gradientMagnitudeTolerance=1e-6)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = registration_method.Execute(fixed_image, moving_image)
            moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform,
                                             sitk.sitkLinear, 0.0, moving_image.GetPixelID())
            self.logger.info("Successfully registered dose image to CT.")
            return moving_resampled
        except Exception as e:
            self.logger.error(f"Error during registration: {e}")
            return None

    def calculate_volume(self, dose_image, dose_threshold_percentage):
        try:
            dose_array = sitk.GetArrayFromImage(dose_image)
            max_dose = dose_array.max()
            if max_dose == 0:
                self.logger.warning("Maximum dose is 0.")
                return 0.0
            dose_threshold = (dose_threshold_percentage / 100.0) * max_dose
            high_dose_voxels = dose_array > dose_threshold
            spacing = dose_image.GetSpacing()
            voxel_volume_cm3 = (spacing[0] / 10.0) * (spacing[1] / 10.0) * (spacing[2] / 10.0)
            total_voxels = np.sum(high_dose_voxels)
            total_volume = total_voxels * voxel_volume_cm3
            self.logger.info(f"Volume above {dose_threshold_percentage}% threshold: {total_volume:.2f} cm³")
            return total_volume
        except Exception as e:
            self.logger.error(f"Error calculating volume: {e}")
            return 0.0

    def load_rtstruct(self, rtstruct_files):
        self.logger.info("Processing RTSTRUCT files for Brainstem only.")
        for rtstruct_file in rtstruct_files:
            try:
                self.logger.debug(f"Processing RTSTRUCT file: {rtstruct_file}")
                rtstruct_data = pydicom.dcmread(rtstruct_file)
                if not (hasattr(rtstruct_data, 'StructureSetROISequence') and hasattr(rtstruct_data, 'ROIContourSequence')):
                    self.logger.warning("RTSTRUCT file missing necessary sequences.")
                    continue
                for roi in rtstruct_data.StructureSetROISequence:
                    raw_name = roi.ROIName
                    canonical_name = canonicalize_roi_name(raw_name)
                    if canonical_name != "Brainstem":
                        continue
                    self.logger.debug(f"Found Brainstem ROI: '{raw_name}' normalized to '{canonical_name}'")
                    ct_array = sitk.GetArrayFromImage(self.ct_image)
                    if "Brainstem" not in self.rtstruct_masks:
                        self.rtstruct_masks["Brainstem"] = np.zeros(ct_array.shape, dtype=np.uint8)
                    for roi_contour in rtstruct_data.ROIContourSequence:
                        if roi_contour.ReferencedROINumber != roi.ROINumber:
                            continue
                        if not hasattr(roi_contour, 'ContourSequence'):
                            continue
                        origin = self.ct_image.GetOrigin()
                        spacing = self.ct_image.GetSpacing()
                        num_slices = ct_array.shape[0]
                        z_positions = np.array([origin[2] + (i + 0.5) * spacing[2] for i in range(num_slices)])
                        tol = 1e-3
                        for contour in roi_contour.ContourSequence:
                            contour_data = contour.ContourData
                            if not contour_data:
                                continue
                            pts = np.array(contour_data).reshape(-1, 3)
                            z_vals = pts[:, 2]
                            if np.ptp(z_vals) < tol:
                                slice_index = int(np.argmin(np.abs(z_positions - z_vals[0])))
                                slices_to_fill = [slice_index]
                            else:
                                min_z = z_vals.min()
                                max_z = z_vals.max()
                                slices_to_fill = np.where((z_positions >= min_z) & (z_positions <= max_z))[0].tolist()
                            pts_xy = pts[:, :2]
                            origin_xy = np.array(origin[:2])
                            spacing_xy = np.array(spacing[:2])
                            pixel_coords = np.floor((pts_xy - origin_xy) / spacing_xy + 0.5).astype(int)
                            for slice_index in slices_to_fill:
                                slice_shape = self.rtstruct_masks["Brainstem"][slice_index].shape
                                rr, cc = polygon(pixel_coords[:, 1], pixel_coords[:, 0], shape=slice_shape)
                                self.rtstruct_masks["Brainstem"][slice_index, rr, cc] = 1
            except Exception as e:
                self.logger.error(f"Error loading RTSTRUCT file {rtstruct_file}: {e}")
        self.logger.info("Finished processing RTSTRUCT files for Brainstem.")
        self.logger.debug("Final RTSTRUCT keys: " + ", ".join(self.rtstruct_masks.keys()))

    def process_patient(self):
        self.logger.info(f"Processing patient {self.patient_id}")
        if not os.path.exists(self.patient_path):
            self.logger.error(f"Patient path not found: {self.patient_path}")
            return
        try:
            studies = [d for d in os.listdir(self.patient_path) if os.path.isdir(os.path.join(self.patient_path, d))]
        except Exception as e:
            self.logger.error(f"Error listing studies in {self.patient_path}: {e}")
            return
        rt_study = next((s for s in studies if "Radiotherapy planning" in s), None)
        if rt_study is None:
            self.logger.error("No radiotherapy planning study found.")
            return
        rt_study_path = os.path.join(self.patient_path, rt_study)
        try:
            series = [d for d in os.listdir(rt_study_path) if os.path.isdir(os.path.join(rt_study_path, d))]
        except Exception as e:
            self.logger.error(f"Error listing series in {rt_study_path}: {e}")
            return
        ct_series_path = None
        rtdose_files = []
        rtdose_series_paths = []
        rtplan_files = []
        rtstruct_files = []
        for s in series:
            if s.startswith('._'):
                continue
            series_path = os.path.join(rt_study_path, s)
            dicom_files = self.find_dicom_files(series_path)
            if not dicom_files:
                continue
            first_dicom = dicom_files[0][1]
            modality = getattr(first_dicom, 'Modality', '')
            if modality == "CT":
                ct_series_path = series_path
            elif modality == "RTDOSE":
                if len(dicom_files) == 1:
                    rtdose_files.append(dicom_files[0][0])
                else:
                    rtdose_series_paths.append(series_path)
            elif modality == "RTPLAN":
                rtplan_files.extend([file for file, _ in dicom_files])
            elif modality == "RTSTRUCT":
                rtstruct_files.extend([file for file, _ in dicom_files])
        if rtplan_files:
            self.process_rtplan(rtplan_files)
        else:
            self.logger.warning("No RTPLAN files found.")
        if ct_series_path is not None:
            self.ct_image = self.load_dicom_series(ct_series_path)
            if self.ct_image is None:
                self.logger.error("CT image could not be loaded.")
                return
        else:
            self.logger.error("No CT series found.")
            return
        for file in rtdose_files:
            self.logger.info(f"Processing RTDOSE file: {file}")
            dose_img = self.load_dicom_file(file)
            if dose_img is None:
                continue
            reg_dose = self.register_images(self.ct_image, dose_img)
            if reg_dose is None:
                continue
            self.registered_doses.append(reg_dose)
            vol = self.calculate_volume(reg_dose, self.dose_threshold_percentage)
            self.logger.info(f"Volume above {self.dose_threshold_percentage}%: {vol:.2f} cm³")
        for series_path in rtdose_series_paths:
            self.logger.info(f"Processing RTDOSE series: {series_path}")
            dose_img = self.load_dicom_series(series_path)
            if dose_img is None:
                continue
            reg_dose = self.register_images(self.ct_image, dose_img)
            if reg_dose is None:
                continue
            self.registered_doses.append(reg_dose)
            vol = self.calculate_volume(reg_dose, self.dose_threshold_percentage)
            self.logger.info(f"Volume above {self.dose_threshold_percentage}%: {vol:.2f} cm³")
        if self.registered_doses:
            self.registered_dose = max(self.registered_doses, key=lambda img: sitk.GetArrayFromImage(img).max())
            self.logger.info("Selected registered dose image with highest maximum dose.")
        else:
            self.logger.error("No registered dose images available.")
        if rtstruct_files:
            self.logger.info("Processing RTSTRUCT files...")
            self.load_rtstruct(rtstruct_files)
        else:
            self.logger.warning("No RTSTRUCT files found.")

    def get_dose_array(self):
        if self.registered_dose is not None:
            try:
                return sitk.GetArrayFromImage(self.registered_dose)
            except Exception as e:
                self.logger.error(f"Error converting registered dose image to array: {e}")
                return None
        else:
            self.logger.error("No registered dose image available.")
            return None


class DoseLoaderCache:
    """
    Caches DoseLoader instances to avoid reloading dose data for patients.
    """
    def __init__(self, max_size=50, base_path="Burdenko-GBM-Progression", dose_threshold_percentage=95.0, verbose=False):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.base_path = base_path
        self.dose_threshold_percentage = dose_threshold_percentage
        self.verbose = verbose

    def get(self, patient_id):
        if patient_id in self.cache:
            self.cache.move_to_end(patient_id)
            return self.cache[patient_id]
        else:
            try:
                loader = DoseLoader(patient_id, self.base_path, self.dose_threshold_percentage, self.verbose)
                self.cache[patient_id] = loader
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                return loader
            except Exception as e:
                if self.verbose:
                    print(f"Error loading DoseLoader for patient {patient_id}: {e}")
                return None

class GBMDataset3D_Survival(Dataset):
    def __init__(self, filtered_data_csv, base_path="Burdenko-GBM-Progression",
                 transform=None, verbose=True, cache_size=10,
                 select_top_slices=True, num_slices=100, target_spacing=(1.0, 1.0, 1.0), label_column='Survived_Over_600_Days'):
        """
        Initializes GBMDataset3D_Survival.
        Parameters:
        - filtered_data_csv (str): Path to the filtered data CSV containing patient IDs, features, and labels.
        - base_path (str): Base directory containing patient data.
        - transform (callable, optional): Optional transformations to be applied to the image data.
        - verbose (bool): If True, enables detailed print statements.
        - cache_size (int): Maximum number of patients to cache.
        - select_top_slices (bool): If True, selects the top contiguous slices with the highest dose.
        - num_slices (int): Number of slices to select if select_top_slices is True.
        - target_spacing (tuple): Desired voxel spacing (z, y, x) in mm.
        """
        self.base_path = base_path
        self.transform = transform
        self.verbose = verbose
        self.cache_size = cache_size
        self.select_top_slices = select_top_slices
        self.num_slices = num_slices
        self.target_shape = (self.num_slices, 256, 256)  # Fast slice-størrelse
        self.target_spacing = target_spacing  # Isotropisk voxel spacing i mm

        # Last inn og prosesser data fra CSV
        self.filtered_data = pd.read_csv(filtered_data_csv)

        # Filtrer pasienter med gyldig 'Survived_Over_600_Days' etikett
        self.filtered_data = self.filtered_data.dropna(subset=['Survived_Over_600_Days'])

        # Konverter etikett til binær
        self.filtered_data['Survived_Over_600_Days'] = self.filtered_data['Survived_Over_600_Days'].astype(int)

        # Definer funksjoner
        self.features = [
            'AgeAtStudyDate', 'Sex', 'IDH1/2', 'MGMT'
        ]

        self.patient_ids = self.filtered_data['Patient_ID'].tolist()
        self.labels = self.filtered_data['Survived_Over_600_Days'].tolist()
        
        # Forbered kliniske funksjoner
        self.additional_features = self.filtered_data[self.features].copy()

        # Håndter manglende verdier
        self._handle_missing_values()

        # Kodifiser kategoriske variabler
        self._encode_categorical_features()

        # Standardiser numeriske funksjoner
        self._standardize_numerical_features()

        # Konverter til numpy array for effektiv tilgang
        self.processed_features = self.additional_features.values.astype(np.float32)

        # Initialiser ProcessorCache (forutsetter at den er definert et annet sted)
        self.processor_cache = ProcessorCache(max_size=self.cache_size, base_path=self.base_path, verbose=self.verbose)

    def _handle_missing_values(self):
        if self.verbose:
            print("Imputation of numerical features...")
        numerical_features = [
            'AgeAtStudyDate'
        ]
        self.additional_features[numerical_features] = self.additional_features[numerical_features].fillna(
            self.additional_features[numerical_features].median()
        )

        categorical_features = ['Sex', 'IDH1/2', 'MGMT']
        self.additional_features[categorical_features] = self.additional_features[categorical_features].fillna('Missing')

    def _encode_categorical_features(self):
        if self.verbose:
            print("One-hot encoding categorical features...")
        categorical_features = ['Sex', 'IDH1/2', 'MGMT']
        self.additional_features = pd.get_dummies(self.additional_features, columns=categorical_features)

    def _standardize_numerical_features(self):
        if self.verbose:
            print("Normalizing numerical features...")
        numerical_features = [
            'AgeAtStudyDate'
        ]
        # Normalize the numerical features
        for feature in numerical_features:
            mean = self.additional_features[feature].mean()
            std = self.additional_features[feature].std()
            if std != 0:
                self.additional_features[feature] = (self.additional_features[feature] - mean) / std
            else:
                self.additional_features[feature] = self.additional_features[feature] - mean  # Hvis std er 0

    def _select_top_connected_slices(self, dose_volume, num_slices=64):
        """
        Selects a contiguous window of slices with the highest total dose.

        Parameters:
        - dose_volume: numpy.ndarray, dose distribution with shape (depth, height, width)
        - num_slices: int, number of contiguous slices to select

        Returns:
        - slice object indicating the range of selected slices
        """
        # Sums dose per slice
        slice_doses = dose_volume.sum(axis=(1, 2))
        total_slices = len(slice_doses)

        if total_slices <= num_slices:
            if self.verbose:
                print(f"Not enough ({total_slices}) to choose {num_slices}. Chose all slices.")
            return slice(0, total_slices)

        # Calculate window sums using convolution to get the slices with highest dose
        window_sums = np.convolve(slice_doses, np.ones(num_slices), mode='valid')
        max_index = np.argmax(window_sums)
        selected_slices = slice(max_index, max_index + num_slices)
        if self.verbose:
            print(f"Chose slices from {selected_slices.start} to {selected_slices.stop - 1} with total dose {window_sums[max_index]:.2f}")
        return selected_slices

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)  # Binær etikett som float

        # Get patient data from cache
        processor = self.processor_cache.get(patient_id)
        original_spacing = processor.original_spacing  # Tuple: (z, y, x)
        if processor is None:
            raise RuntimeError(f"Feilet å laste data for pasient {patient_id}")
        if processor.mr_array is None or processor.dose_array is None or processor.mr_t1_array is None:
            processor.load_all_data()
        if processor.mr_array is None or processor.mr_t1_array is None:
            raise RuntimeError(f"Data ikke korrekt lastet for pasient {patient_id}")

        # Get image volumes
        mr_t2_volume = processor.registered_mr_array.astype(np.float32)  # MR T2 volum
        mr_t1_volume = processor.registered_mr_t1_array.astype(np.float32)  # MR T1 volum
        dose_volume = processor.dose_array.astype(np.float32) if processor.dose_array is not None else np.zeros_like(mr_t2_volume)

        if self.select_top_slices:
            selected_slices = self._select_top_connected_slices(dose_volume, self.num_slices)
            mr_t2_volume = mr_t2_volume[selected_slices, :, :]
            mr_t1_volume = mr_t1_volume[selected_slices, :, :]
            dose_volume = dose_volume[selected_slices, :, :]
        else:
            # Resize the volumes to target shape
            mr_t2_volume = resize(mr_t2_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
            mr_t1_volume = resize(mr_t1_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
            dose_volume = resize(dose_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
        
        
        mr_t2_volume = resample_image(mr_t2_volume, original_spacing, self.target_spacing, self.target_shape)
        mr_t1_volume = resample_image(mr_t1_volume, original_spacing, self.target_spacing, self.target_shape)
        dose_volume = resample_image(dose_volume, original_spacing, self.target_spacing, self.target_shape)
        # Normalize the image volumes 
        mr_t2_mean = mr_t2_volume.mean()
        mr_t2_std = mr_t2_volume.std()
        mr_t2_volume = (mr_t2_volume - mr_t2_mean) / (mr_t2_std + 1e-8)

        mr_t1_mean = mr_t1_volume.mean()
        mr_t1_std = mr_t1_volume.std()
        mr_t1_volume = (mr_t1_volume - mr_t1_mean) / (mr_t1_std + 1e-8)

        dose_mean = dose_volume.mean()
        dose_std = dose_volume.std()
        #dose_volume = (dose_volume - dose_mean) / (dose_std + 1e-8)

        # Stack the volumes to create the input image
        image = np.stack([mr_t1_volume, mr_t2_volume, dose_volume], axis=0)  # Shape: (3, num_slices, 256, 256)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        # Get clinical features
        clinical_features = torch.tensor(self.processed_features[idx], dtype=torch.float32)

        sample = {
            'image': image,  # Tensor: (3, num_slices, 256, 256)
            'label': label,  # Tensor: ()
            'clinical_features': clinical_features,  # Tensor: (num_features,)
            'patient_id': torch.tensor(patient_id, dtype=torch.long)
        }
        # Validering av konsistens
        assert image.shape == torch.Size([3, self.num_slices, 256, 256]), \
            f"Inconsistent image shape for pasient {patient_id}: {image.shape}"

        return sample
    
class GBMDataset3D_PFS(Dataset):
    def __init__(self, filtered_data_csv, base_path="Burdenko-GBM-Progression",
                 transform=None, verbose=True, cache_size=10,
                 select_top_slices=True, num_slices=100, target_spacing=(1.0, 1.0, 1.0)):
        """
        Initializes GBMDataset3D_Survival.
        Parameters:
        - filtered_data_csv (str): Path to the filtered data CSV containing patient IDs, features, and labels.
        - base_path (str): Base directory containing patient data.
        - transform (callable, optional): Optional transformations to be applied to the image data.
        - verbose (bool): If True, enables detailed print statements.
        - cache_size (int): Maximum number of patients to cache.
        - select_top_slices (bool): If True, selects the top contiguous slices with the highest dose.
        - num_slices (int): Number of slices to select if select_top_slices is True.
        - target_spacing (tuple): Desired voxel spacing (z, y, x) in mm.
        """
        self.base_path = base_path
        self.transform = transform
        self.verbose = verbose
        self.cache_size = cache_size
        self.select_top_slices = select_top_slices
        self.num_slices = num_slices
        self.target_shape = (self.num_slices, 256, 256)  # Fast slice-størrelse
        self.target_spacing = target_spacing  # Isotropisk voxel spacing i mm

        # Last inn og prosesser data fra CSV
        self.filtered_data = filtered_data_csv

        # Definer funksjoner
        self.features = [
            'AgeAtStudyDate', 'Sex', 'IDH1/2', 'MGMT'
        ]

        # Separér pasient-IDer og etiketter
        self.patient_ids = filtered_data_csv['AnonymPatientID'].tolist()
        self.pfs_months = filtered_data_csv['PFS_months'].tolist()
        self.pfs_event = filtered_data_csv['PFS_event'].tolist()

        # Initialiser DoseLoaderCache
        self.dose_loader_cache = DoseLoaderCache(
            max_size=self.cache_size, 
            base_path=self.base_path, 
            dose_threshold_percentage=95.0,  # Juster etter behov
            verbose=self.verbose
        )
        # Forbered kliniske funksjoner
        self.additional_features = self.filtered_data[self.features].copy()

        # Håndter manglende verdier
        self._handle_missing_values()

        # Kodifiser kategoriske variabler
        self._encode_categorical_features()

        # Standardiser numeriske funksjoner
        self._standardize_numerical_features()

        # Konverter til numpy array for effektiv tilgang
        self.processed_features = self.additional_features.values.astype(np.float32)

        # Initialiser ProcessorCache (forutsetter at den er definert et annet sted)
        self.processor_cache = ProcessorCache(max_size=self.cache_size, base_path=self.base_path, verbose=self.verbose)

    def _handle_missing_values(self):
        if self.verbose:
            print("Imputation of numerical features...")
        numerical_features = [
            'AgeAtStudyDate'
        ]
        self.additional_features[numerical_features] = self.additional_features[numerical_features].fillna(
            self.additional_features[numerical_features].median()
        )

        categorical_features = ['Sex', 'IDH1/2', 'MGMT']
        self.additional_features[categorical_features] = self.additional_features[categorical_features].fillna('Missing')

    def _encode_categorical_features(self):
        if self.verbose:
            print("One-hot encoding categorical features...")
        categorical_features = ['Sex', 'IDH1/2', 'MGMT']
        self.additional_features = pd.get_dummies(self.additional_features, columns=categorical_features)

    def _standardize_numerical_features(self):
        if self.verbose:
            print("Normalizing numerical features...")
        numerical_features = [
            'AgeAtStudyDate'
        ]
        # Normalize the numerical features
        for feature in numerical_features:
            mean = self.additional_features[feature].mean()
            std = self.additional_features[feature].std()
            if std != 0:
                self.additional_features[feature] = (self.additional_features[feature] - mean) / std
            else:
                self.additional_features[feature] = self.additional_features[feature] - mean  # Hvis std er 0

    def _select_top_connected_slices(self, dose_volume, num_slices=100):
        """
        Selects a contiguous window of slices with the highest total dose.

        Parameters:
        - dose_volume: numpy.ndarray, dose distribution with shape (depth, height, width)
        - num_slices: int, number of contiguous slices to select

        Returns:
        - slice object indicating the range of selected slices
        """
        # Sums dose per slice
        slice_doses = dose_volume.sum(axis=(1, 2))
        total_slices = len(slice_doses)

        if total_slices <= num_slices:
            if self.verbose:
                print(f"Not enough ({total_slices}) to choose {num_slices}. Chose all slices.")
            return slice(0, total_slices)

        # Calculate window sums using convolution to get the slices with highest dose
        window_sums = np.convolve(slice_doses, np.ones(num_slices), mode='valid')
        max_index = np.argmax(window_sums)
        selected_slices = slice(max_index, max_index + num_slices)
        if self.verbose:
            print(f"Chose slices from {selected_slices.start} to {selected_slices.stop - 1} with total dose {window_sums[max_index]:.2f}")
        return selected_slices

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        pfs_months = self.pfs_months[idx]  # Bruk PFS måneder direkte
        pfs_event = self.pfs_event[idx]
        # Konverter pasient-ID til numerisk verdi for tensor
        if isinstance(patient_id, str) and 'Burdenko-GBM-' in patient_id:
            patient_num = int(patient_id.split('-')[-1])
        else:
            patient_num = int(patient_id)

        # Get patient data from cache
        processor = self.processor_cache.get(patient_id)
        original_spacing = processor.original_spacing  # Tuple: (z, y, x)
        if processor is None:
            raise RuntimeError(f"Feilet å laste data for pasient {patient_id}")
        if processor.mr_array is None or processor.dose_array is None or processor.mr_t1_array is None:
            processor.load_all_data()
        if processor.mr_array is None or processor.mr_t1_array is None:
            raise RuntimeError(f"Data ikke korrekt lastet for pasient {patient_id}")

        # Get image volumes
        mr_t2_volume = processor.registered_mr_array.astype(np.float32)  # MR T2 volum
        mr_t1_volume = processor.registered_mr_t1_array.astype(np.float32)  # MR T1 volum
        dose_volume = processor.dose_array.astype(np.float32) if processor.dose_array is not None else np.zeros_like(mr_t2_volume)

        if self.select_top_slices:
            selected_slices = self._select_top_connected_slices(dose_volume, self.num_slices)
            mr_t2_volume = mr_t2_volume[selected_slices, :, :]
            mr_t1_volume = mr_t1_volume[selected_slices, :, :]
            dose_volume = dose_volume[selected_slices, :, :]
        else:
            # Resize the volumes to target shape
            mr_t2_volume = resize(mr_t2_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
            mr_t1_volume = resize(mr_t1_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
            dose_volume = resize(dose_volume, self.target_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
        
        
        mr_t2_volume = resample_image(mr_t2_volume, original_spacing, self.target_spacing, self.target_shape)
        mr_t1_volume = resample_image(mr_t1_volume, original_spacing, self.target_spacing, self.target_shape)
        dose_volume = resample_image(dose_volume, original_spacing, self.target_spacing, self.target_shape)
        # Normalize the image volumes 
        mr_t2_mean = mr_t2_volume.mean()
        mr_t2_std = mr_t2_volume.std()
        mr_t2_volume = (mr_t2_volume - mr_t2_mean) / (mr_t2_std + 1e-9)

        mr_t1_mean = mr_t1_volume.mean()
        mr_t1_std = mr_t1_volume.std()
        mr_t1_volume = (mr_t1_volume - mr_t1_mean) / (mr_t1_std + 1e-9)

        dose_mean = dose_volume.mean()
        dose_std = dose_volume.std()
        dose_volume = (dose_volume - dose_mean) / (dose_std + 1e-9)

        # Hent dose-relaterte variabler fra DoseLoaderCache
        dose_loader = self.dose_loader_cache.get(patient_id)
        if dose_loader is None:
            # Default values if DoseLoader fails
            prescription_dose = -1.0
            number_of_fractions_planned = -1
            delivery_maximum_dose = -1.0
            volume_above_95_percent_dose = -1.0
        else:
            # Retrieve dose-related features from DoseLoader
            prescription_dose = dose_loader.prescription_dose
            number_of_fractions_planned = dose_loader.number_of_fractions_planned
            delivery_maximum_dose = dose_loader.delivery_maximum_dose
            volume_above_95_percent_dose = getattr(dose_loader, 'volume_above_threshold', -1.0)

        # Combine dose-related features into a tensor
        dose_features = torch.tensor([
            prescription_dose,
            number_of_fractions_planned,
            delivery_maximum_dose,
            volume_above_95_percent_dose
        ], dtype=torch.float32)
        # Initialiser DoseLoaderCache
        # Stack the volumes to create the input image
        image = np.stack([mr_t1_volume, mr_t2_volume, dose_volume], axis=0)  # Shape: (3, num_slices, 256, 256)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        # Get clinical features
        clinical_features = torch.tensor(self.processed_features[idx], dtype=torch.float32)
        combined_features = torch.cat([clinical_features, torch.tensor(dose_features)], dim=0)
        sample = {
            'image': image,  # Tensor: (3, num_slices, 256, 256)
            'clinical_features': combined_features,  # Tensor: (num_features,)
            'patient_id': torch.tensor(patient_num, dtype=torch.long),
            'pfs_months': torch.tensor(pfs_months, dtype=torch.float32),
            'pfs_event': torch.tensor(pfs_event, dtype=torch.long)
        }
        # Validering av konsistens
        assert image.shape == torch.Size([3, self.num_slices, 256, 256]), \
            f"Inconsistent image shape for pasient {patient_id}: {image.shape}"

        return sample

class Plotter:
    def __init__(self, processor, verbose=False, target_resolution=None):
        """
        Initialiserer Plotter med en DICOMProcessor-instans.
        
        Parameters:
          - processor (DICOMProcessor): En instans av DICOMProcessor som inneholder bildedata.
          - verbose (bool): Hvis True, viser detaljerte utskrifter.
          - target_resolution (tuple or None): Hvis satt til (width, height) (f.eks. (256,256)), 
            vil hver slice bli resized før plotting.
        """
        self.processor = processor
        self.verbose = verbose
        self.target_resolution = target_resolution
    
    def _resize_if_needed(self, image):
        """Hjelpefunksjon for å resize bildet dersom target_resolution er satt."""
        if self.target_resolution is not None:
            # Check if image is valid and non-empty
            if image is None or not hasattr(image, 'shape') or image.size == 0:
                if self.verbose:
                    print("Warning: image is empty or invalid, skipping resizing.")
                return image
            try:
                # Ensure target_resolution is in the expected (width, height) format as integers.
                target_res = (int(self.target_resolution[0]), int(self.target_resolution[1]))
                return cv2.resize(image, target_res, interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                if self.verbose:
                    print("Error during resizing:", e)
                return image
        return image

    def plot_ct_slice(self, slice_idx, ax=None):
        """
        Plotter CT-slice ved spesifisert indeks.
        
        Parameters:
          - slice_idx (int): Indeksen til slice som skal plottes.
          - ax (matplotlib.axes.Axes, optional): Akse hvor plottet skal tegnes. Hvis None, opprettes en ny akse.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
        
        ct = self.processor.ct_array
        if ct is not None:
            if 0 <= slice_idx < ct.shape[0]:
                img = ct[slice_idx, :, :]
                img = self._resize_if_needed(img)
                ax.imshow(img, cmap='gray')
                if self.verbose:
                    print(f"Plottet CT slice {slice_idx} med shape {img.shape}")
            else:
                ax.text(0.5, 0.5, f'CT slice {slice_idx} out of range', 
                        horizontalalignment='center', verticalalignment='center')
                if self.verbose:
                    print(f"CT slice {slice_idx} er utenfor rekkevidden ({ct.shape[0]})")
        else:
            ax.text(0.5, 0.5, 'CT not available', 
                    horizontalalignment='center', verticalalignment='center')
            if self.verbose:
                print("CT array er None")
        ax.axis('off')
    
    def plot_mr_t1_slice(self, slice_idx, ax=None):
        """
        Plotter MR T1-slice ved spesifisert indeks.
        
        Parameters:
          - slice_idx (int): Indeksen til slice som skal plottes.
          - ax (matplotlib.axes.Axes, optional): Akse hvor plottet skal tegnes. Hvis None, opprettes en ny akse.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
        
        mr_t1 = self.processor.registered_mr_t1_array
        if mr_t1 is not None:
            if 0 <= slice_idx < mr_t1.shape[0]:
                img = mr_t1[slice_idx, :, :]
                img = self._resize_if_needed(img)
                ax.imshow(img, cmap='gray')
                if self.verbose:
                    print(f"Plottet MR T1 slice {slice_idx} med shape {img.shape}")
            else:
                ax.text(0.5, 0.5, f'MR T1 slice {slice_idx} out of range', 
                        horizontalalignment='center', verticalalignment='center')
                if self.verbose:
                    print(f"MR T1 slice {slice_idx} er utenfor rekkevidden ({mr_t1.shape[0]})")
        else:
            ax.text(0.5, 0.5, 'MR T1 not available', 
                    horizontalalignment='center', verticalalignment='center')
            if self.verbose:
                print("MR T1 array er None")
        ax.axis('off')
    
    def plot_mr_t2_slice(self, slice_idx, ax=None):
        """
        Plotter MR T2-slice ved spesifisert indeks.
        
        Parameters:
          - slice_idx (int): Indeksen til slice som skal plottes.
          - ax (matplotlib.axes.Axes, optional): Akse hvor plottet skal tegnes. Hvis None, opprettes en ny akse.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
        
        mr_t2 = self.processor.registered_mr_array
        if mr_t2 is not None:
            if 0 <= slice_idx < mr_t2.shape[0]:
                img = mr_t2[slice_idx, :, :]
                img = self._resize_if_needed(img)
                ax.imshow(img, cmap='gray')
                if self.verbose:
                    print(f"Plottet MR T2 slice {slice_idx} med shape {img.shape}")
            else:
                ax.text(0.5, 0.5, f'MR T2 slice {slice_idx} out of range', 
                        horizontalalignment='center', verticalalignment='center')
                if self.verbose:
                    print(f"MR T2 slice {slice_idx} er utenfor rekkevidden ({mr_t2.shape[0]})")
        else:
            ax.text(0.5, 0.5, 'MR T2 not available', 
                    horizontalalignment='center', verticalalignment='center')
            if self.verbose:
                print("MR T2 array er None")
        ax.axis('off')
    
    def plot_dose_slice(self, slice_idx, ax=None):
        """
        Plots Dose slice at the specified index with correct scaling.
        
        Parameters:
          - slice_idx (int): Index of the slice to plot.
          - ax (matplotlib.axes.Axes, optional): Axes where the plot will be drawn. If None, a new axes is created.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
        dose = self.processor.dose_array
        dose_grid_scaling = getattr(self.processor, 'dose_grid_scaling', 1.0)

        if dose is not None:
            if 0 <= slice_idx < dose.shape[0]:
                dose_scaled = dose[slice_idx, :, :] * dose_grid_scaling
                # Optionally resize the dose slice
                dose_scaled = self._resize_if_needed(dose_scaled)
                im = ax.imshow(dose_scaled, cmap='jet')
                dose_units = getattr(self.processor, 'dose_units', 'Dose (Gy)')
                cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(dose_units)
                
                tick_min = np.floor(dose_scaled.min() / 10) * 10
                tick_max = np.ceil(dose_scaled.max() / 10) * 10
                ticks = np.arange(tick_min, tick_max, 10)
                cbar.set_ticks(ticks)
                
                if self.verbose:
                    print(f"Plotted Dose slice {slice_idx} with shape {dose_scaled.shape}")
            else:
                ax.text(0.5, 0.5, f'Dose slice {slice_idx} out of range', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                if self.verbose:
                    print(f"Dose slice {slice_idx} is out of range ({dose.shape[0]})")
        else:
            ax.text(0.5, 0.5, 'Dose not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            if self.verbose:
                print("Dose array is None")
        ax.axis('off')
