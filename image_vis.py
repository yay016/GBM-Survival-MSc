"""
GBM Image Processing and Interactive Visualization Script

This script provides an interactive environment for processing and visualizing multimodal DICOM images,
including CT, MR (with bias correction and registration), and RTDOSE data.
It offers control over image registration, bias field correction, and cropping operations,
as well as interactive slice navigation and structure visualization using a modern Tkinter-based interface.
Users can select the patient and MR modality via dropdown menus and adjust visualization parameters with
buttons and sliders. The script is designed for clinical research and quality control of imaging data.
"""

import os
import numpy as np
import matplotlib
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import SimpleITK as sitk
from collections import defaultdict

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use TkAgg backend for interactive matplotlib in Tkinter
matplotlib.use('TkAgg')

# =============================================================================
# DICOM Loader and Image Processing Classes
# =============================================================================

class DICOMLoader:
    """
    Loads and organizes DICOM files for a given patient, including CT, MR, RTSTRUCT, RTPLAN, and RTDOSE data.
    """
    def __init__(self, patient_id, base_path="Burdenko-GBM-Progression"):
        self.patient_id = patient_id
        self.patient_file_name = f"Burdenko-GBM-{patient_id:03d}"
        self.patient_path = os.path.join(base_path, self.patient_file_name)
        self.studies = []
        self.series = []
        self.series_dict = {}
        self.rtstruct_files = []
        self.rtplan_files = []
        self.rtdose_files = []
        self.rtdose_series_paths = []

    def find_dicom_files(self, series_path):
        files = os.listdir(series_path)
        dicom_files = []
        for file in files:
            if file.startswith('._'):
                continue  # Ignore files starting with ._
            if file.endswith(".dcm"):
                filepath = os.path.join(series_path, file)
                try:
                    dicom_data = pydicom.dcmread(filepath, force=True)
                    dicom_files.append((filepath, dicom_data))
                except (pydicom.errors.InvalidDicomError, AttributeError):
                    print(f"Invalid or incomplete DICOM file skipped: {filepath}")
        return dicom_files

    def load_studies(self):
        if not os.path.exists(self.patient_path):
            raise FileNotFoundError(f"Patient path not found: {self.patient_path}")
        self.studies = [d for d in os.listdir(self.patient_path) if os.path.isdir(os.path.join(self.patient_path, d))]
        radiotherapy_study = None
        for study in self.studies:
            if "Radiotherapy planning" in study:
                radiotherapy_study = study
                print(f"Radiotherapy Study Found: {study}")
                break
        if radiotherapy_study is None:
            raise FileNotFoundError("No radiotherapy study found for this patient.")
        self.rt_study_path = os.path.join(self.patient_path, radiotherapy_study)
        self.series = [d for d in os.listdir(self.rt_study_path) if os.path.isdir(os.path.join(self.rt_study_path, d))]
        print("Series found in Radiotherapy Study:")
        for s in self.series:
            print(s)

    def load_series(self):
        for s in self.series:
            if s.startswith('._'):
                continue
            series_path = os.path.join(self.rt_study_path, s)
            dicom_files_in_series = self.find_dicom_files(series_path)
            if not dicom_files_in_series:
                continue
            first_dicom_file = dicom_files_in_series[0][1]
            modality = getattr(first_dicom_file, 'Modality', '')
            series_description = getattr(first_dicom_file, 'SeriesDescription', '')
            if modality == "CT":
                self.series_dict['CT'] = series_path
            elif modality == "MR":
                # Decide MR modality based on series description
                if "T1CE" in series_description or "CET1" in series_description:
                    self.series_dict['MR_T1CE'] = series_path
                elif "T2FLAIR" in series_description or "T2 FLAIR" in series_description:
                    self.series_dict['MR_T2FLAIR'] = series_path
            elif modality == "RTSTRUCT":
                self.rtstruct_files.extend(dicom_files_in_series)
            elif modality == "RTPLAN":
                self.rtplan_files.extend(dicom_files_in_series)
            elif modality == "RTDOSE":
                if len(dicom_files_in_series) == 1:
                    self.rtdose_files.append(dicom_files_in_series[0])
                else:
                    self.rtdose_series_paths.append(series_path)

    def validate_series(self):
        required_series = ['CT']
        for series_name in required_series:
            if series_name not in self.series_dict:
                raise FileNotFoundError(f"No {series_name} series found.")
            else:
                print(f"{series_name} series found at: {self.series_dict[series_name]}")
        if not self.rtstruct_files:
            raise FileNotFoundError("No RTSTRUCT files found.")
        else:
            print(f"{len(self.rtstruct_files)} RTSTRUCT file(s) found.")
        if not self.rtplan_files:
            raise FileNotFoundError("No RTPLAN files found.")
        else:
            print(f"{len(self.rtplan_files)} RTPLAN file(s) found.")
        if not (self.rtdose_files or self.rtdose_series_paths):
            raise FileNotFoundError("No RTDOSE data found.")
        else:
            print("RTDOSE data found.")

    def load_dicom_series(self, series_path):
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(series_path)
        if not series_IDs:
            print(f"ERROR: No DICOM series found in directory {series_path}")
            return None
        series_file_names = reader.GetGDCMSeriesFileNames(series_path, series_IDs[0])
        reader.SetFileNames(series_file_names)
        try:
            sitk_image = reader.Execute()
            return sitk_image
        except Exception as e:
            print(f"Error loading DICOM series from {series_path}: {e}")
            return None

    def load_rtdose(self):
        if self.rtdose_files:
            rtdose_file_path, rtdose_dataset = self.rtdose_files[0]
            try:
                dose_image_sitk = sitk.ReadImage(rtdose_file_path)
                return dose_image_sitk, rtdose_dataset
            except Exception as e:
                print(f"Error loading RTDOSE file {rtdose_file_path}: {e}")
                return None, None
        elif self.rtdose_series_paths:
            rtdose_series_path = self.rtdose_series_paths[0]
            dose_image_sitk = self.load_dicom_series(rtdose_series_path)
            if dose_image_sitk is None:
                print(f"Could not load RTDOSE series from {rtdose_series_path}")
                return None, None
            rtdose_files_in_series = self.find_dicom_files(rtdose_series_path)
            if not rtdose_files_in_series:
                print("No DICOM files found in RTDOSE series.")
                return dose_image_sitk, None
            rtdose_dataset = rtdose_files_in_series[0][1]
            return dose_image_sitk, rtdose_dataset
        else:
            print("No RTDOSE data available.")
            return None, None

# -----------------------------------------------------------------------------

class ImageProcessor:
    """
    Contains static methods for image processing operations including image registration,
    bias correction, brain masking, resampling, and cropping adjustments.
    """
    def __init__(self):
        pass

    @staticmethod
    def register_images(fixed_image, moving_image):
        fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving = sitk.Cast(moving_image, sitk.sitkFloat32)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-6, numberOfIterations=500,
            gradientMagnitudeTolerance=1e-6)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = registration_method.Execute(fixed, moving)
        registered_moving = sitk.Resample(
            moving, fixed, final_transform, sitk.sitkLinear, 0.0,
            moving.GetPixelID())
        return registered_moving, final_transform

    @staticmethod
    def bias_correction(mr_image_sitk):
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)
        shrink_factor = 2
        maskImage = sitk.OtsuThreshold(mr_image_sitk, 0, 1, 200)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
        mr_down = sitk.Shrink(mr_image_sitk, [shrink_factor]*mr_image_sitk.GetDimension())
        mask_down = sitk.Shrink(maskImage, [shrink_factor]*maskImage.GetDimension())
        mr_corrected_down = corrector.Execute(mr_down, mask_down)
        log_bias_down = corrector.GetLogBiasFieldAsImage(mr_down)
        log_bias_up = sitk.Resample(
            log_bias_down, mr_image_sitk, sitk.Transform(),
            sitk.sitkBSpline, 0.0, log_bias_down.GetPixelID())
        mr_corrected = mr_image_sitk / sitk.Exp(log_bias_up)
        return mr_corrected

    @staticmethod
    def create_brain_mask(registered_mr_sitk):
        brain_mask = sitk.OtsuThreshold(registered_mr_sitk, 0, 1, 200)
        brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
        brain_mask = sitk.BinaryMorphologicalClosing(brain_mask, (2,2,2))
        connected_component_filter = sitk.ConnectedComponentImageFilter()
        brain_cc = connected_component_filter.Execute(brain_mask)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(brain_cc)
        largest_label = None
        largest_size = 0
        for label in label_shape_filter.GetLabels():
            size = label_shape_filter.GetPhysicalSize(label)
            if size > largest_size:
                largest_size = size
                largest_label = label
        brain_mask = sitk.BinaryThreshold(
            brain_cc, lowerThreshold=largest_label, upperThreshold=largest_label,
            insideValue=1, outsideValue=0)
        return brain_mask

    @staticmethod
    def apply_brain_mask(ct_image_sitk, mr_image_sitk, dose_image_sitk, brain_mask):
        brain_mask_ct = sitk.Cast(brain_mask, ct_image_sitk.GetPixelID())
        masked_ct = ct_image_sitk * brain_mask_ct
        brain_mask_mr = sitk.Cast(brain_mask, mr_image_sitk.GetPixelID())
        masked_mr = mr_image_sitk * brain_mask_mr
        brain_mask_dose = sitk.Cast(brain_mask, dose_image_sitk.GetPixelID())
        masked_dose = dose_image_sitk * brain_mask_dose
        return masked_ct, masked_mr, masked_dose

    @staticmethod
    def adjust_origin_after_cropping(original_image, cropped_image, index):
        original_origin = np.array(original_image.GetOrigin())
        spacing = np.array(original_image.GetSpacing())
        direction = np.array(original_image.GetDirection()).reshape(3, 3)
        offset = np.dot(index, spacing * direction)
        new_origin = original_origin + offset
        cropped_image.SetOrigin(new_origin.tolist())
        return cropped_image

    @staticmethod
    def resample_image(reference_image, moving_image, transform=None, interpolator=sitk.sitkBSpline):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0.0)
        if transform:
            resampler.SetTransform(transform)
        else:
            resampler.SetTransform(sitk.Transform())
        try:
            resampled_image = resampler.Execute(moving_image)
            return resampled_image
        except Exception as e:
            print(f"Error during image resampling: {e}")
            return None

# -----------------------------------------------------------------------------

class RTStructProcessor:
    """
    Processes RTSTRUCT data to map anatomical structure contours onto CT slices.
    """
    def __init__(self, rtstruct_datasets):
        self.rtstruct_datasets = rtstruct_datasets
        self.slice_contours = defaultdict(list)
        self.rois_without_contours = set()
        self.all_structure_names = set()

    def map_contours_to_slices(self, ct_slice_z_positions):
        for rtstruct in self.rtstruct_datasets:
            if not hasattr(rtstruct, 'StructureSetROISequence'):
                print("RTSTRUCT data missing 'StructureSetROISequence'. Skipping this dataset.")
                continue
            roi_dict = {}
            for roi in rtstruct.StructureSetROISequence:
                if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                    roi_dict[roi.ROINumber] = roi.ROIName
                else:
                    print("ROI missing 'ROINumber' or 'ROIName'.")
                    continue
            roi_numbers_with_contours = set()
            if hasattr(rtstruct, 'ROIContourSequence'):
                for roi_contour in rtstruct.ROIContourSequence:
                    if hasattr(roi_contour, 'ReferencedROINumber'):
                        roi_numbers_with_contours.add(roi_contour.ReferencedROINumber)
            else:
                print("RTSTRUCT data missing 'ROIContourSequence'. Skipping this dataset.")
                continue
            for roi_number, roi_name in roi_dict.items():
                if roi_number in roi_numbers_with_contours:
                    for roi_contour in rtstruct.ROIContourSequence:
                        if (hasattr(roi_contour, 'ReferencedROINumber') and
                            roi_contour.ReferencedROINumber == roi_number):
                            self.all_structure_names.add(roi_name)
                            if hasattr(roi_contour, 'ContourSequence'):
                                for contour in roi_contour.ContourSequence:
                                    contour_data = getattr(contour, 'ContourData', [])
                                    if not contour_data:
                                        print(f"Skipping contour with empty ContourData for structure '{roi_name}'")
                                        continue
                                    try:
                                        contour_points = np.array(contour_data).reshape(-1, 3)
                                    except ValueError:
                                        print(f"Invalid ContourData format for structure '{roi_name}'")
                                        continue
                                    mean_z = np.mean(contour_points[:, 2])
                                    slice_idx = self.find_matching_slice(mean_z, ct_slice_z_positions)
                                    if slice_idx is not None:
                                        self.slice_contours[slice_idx].append({
                                            'points': contour_points,
                                            'structure_name': roi_name
                                        })
                                        print(f"Mapped structure '{roi_name}' to slice {slice_idx} (Z={mean_z:.2f} mm)")
                                    else:
                                        print(f"Could not map structure '{roi_name}' to any slice.")
                else:
                    self.rois_without_contours.add(roi_name)

    @staticmethod
    def find_matching_slice(mean_z, ct_slice_z_positions, tolerance=1e-3):
        for idx, z in enumerate(ct_slice_z_positions):
            if np.isclose(z, mean_z, atol=tolerance):
                return idx
        return None

# =============================================================================
# Interactive Viewer
# =============================================================================

class InteractiveViewer:
    """
    Provides an interactive interface using matplotlib and Tkinter to visualize CT, MR, and RTDOSE images,
    along with overlaid anatomical structure contours. Users can toggle visibility of different modalities,
    adjust bias correction and cropping, and navigate through image slices.
    """
    def __init__(self, ct_original_array, ct_processed_array,
                 dose_original_array, dose_processed_array,
                 mr_original_array, mr_processed_array,
                 slice_contours_original, slice_contours_cropped,
                 rois_without_contours,
                 ct_original_sitk, ct_processed_sitk,
                 dose_original_sitk, dose_processed_sitk,
                 mr_original_sitk, mr_processed_sitk,
                 bounding_box,
                 apply_cropping, apply_bias_correction,
                 all_structure_names,
                 patient_id, mr_modality):
        self.ct_original_array = ct_original_array
        self.ct_processed_array = ct_processed_array
        self.dose_original_array = dose_original_array
        self.dose_processed_array = dose_processed_array
        self.mr_original_array = mr_original_array
        self.mr_processed_array = mr_processed_array
        self.slice_contours_original = slice_contours_original
        self.slice_contours_cropped = slice_contours_cropped
        self.rois_without_contours = rois_without_contours
        self.all_structure_names = all_structure_names

        self.ct_original_sitk = ct_original_sitk
        self.ct_processed_sitk = ct_processed_sitk
        self.dose_original_sitk = dose_original_sitk
        self.dose_processed_sitk = dose_processed_sitk
        self.mr_original_sitk = mr_original_sitk
        self.mr_processed_sitk = mr_processed_sitk

        self.bounding_box = bounding_box
        self.apply_cropping = apply_cropping
        self.apply_bias_correction = apply_bias_correction

        self.patient_id = patient_id
        self.mr_modality = mr_modality

        # Visibility flags
        self.ct_visible = True
        self.mr_visible = True
        self.dose_visible = True
        self.structures_visible = True
        self.bias_correction_applied = apply_bias_correction
        self.cropping_applied = apply_cropping

        self.unique_structures = sorted(list(all_structure_names))
        cmap = plt.get_cmap('tab20')
        self.structure_colors = {name: cmap(i % cmap.N) for i, name in enumerate(self.unique_structures)}

    def physical_to_pixel_coords(self, contour_points, sitk_image):
        image_origin = np.array(sitk_image.GetOrigin())
        image_spacing = np.array(sitk_image.GetSpacing())
        image_direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        delta = contour_points - image_origin
        index_coords = np.dot(delta, np.linalg.inv(image_direction)) / image_spacing
        pixel_coords = index_coords[:, [1, 0]]
        return pixel_coords

    def launch(self):
        # Select which arrays to display based on cropping and bias correction flags
        ct_array = self.ct_processed_array if self.cropping_applied else self.ct_original_array
        mr_array = self.mr_processed_array if self.bias_correction_applied else self.mr_original_array
        dose_array = self.dose_processed_array if self.cropping_applied else self.dose_original_array

        num_slices = ct_array.shape[0]
        self.current_slice = num_slices // 2

        # Create a larger, modern figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 12))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.20)

        ct_slice = ct_array[self.current_slice, :, :]
        mr_slice = mr_array[self.current_slice, :, :]
        dose_slice = dose_array[self.current_slice, :, :]
        structures_on_slice = self.get_structures_on_slice(self.current_slice)

        self.ct_image_obj = self.ax.imshow(ct_slice, cmap='gray', interpolation='none', origin='upper', alpha=0.5)
        self.mr_image_obj = self.ax.imshow(mr_slice, cmap='gray', interpolation='none', origin='upper', alpha=0.5)
        if np.max(dose_array) > 0:
            self.dose_image_obj = self.ax.imshow(dose_slice, cmap='jet', interpolation='none',
                                                  origin='upper', alpha=0.5, vmin=0, vmax=np.percentile(dose_array, 99))
        else:
            self.dose_image_obj = self.ax.imshow(dose_slice, cmap='jet', interpolation='none', origin='upper', alpha=0.0)

        self.structure_lines = []
        for structure in structures_on_slice:
            contour_points = structure['pixel_coords']
            structure_name = structure['structure_name']
            if contour_points.size == 0:
                continue
            color = self.structure_colors.get(structure_name, 'white')
            line, = self.ax.plot(contour_points[:, 1], contour_points[:, 0],
                                 linewidth=2, label=structure_name, color=color)
            self.structure_lines.append(line)

        self.annotation_texts = []
        if self.rois_without_contours:
            text_y = 0.95
            for roi_name in self.rois_without_contours:
                text_obj = self.ax.text(0.05, text_y, f"No contour data for ROI: {roi_name}",
                                        transform=self.ax.transAxes, fontsize=10, color='red')
                self.annotation_texts.append(text_obj)
                text_y -= 0.03

        self.legend_obj = None
        if self.structure_lines:
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.legend_obj = self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

        self.ax.set_title(f"Patient: {self.patient_id}, {self.mr_modality} - Slice {self.current_slice}", fontsize=16)

        self.add_controls(num_slices)
        return self.fig

    def get_structures_on_slice(self, slice_idx):
        if self.cropping_applied:
            slice_contours = self.slice_contours_cropped
            ct_sitk = self.ct_processed_sitk
        else:
            slice_contours = self.slice_contours_original
            ct_sitk = self.ct_original_sitk

        structures = []
        if slice_idx in slice_contours:
            for contour_info in slice_contours[slice_idx]:
                contour_points = contour_info['points']
                structure_name = contour_info['structure_name']
                pixel_coords = self.physical_to_pixel_coords(contour_points, ct_sitk)
                rows, cols = ct_sitk.GetSize()[1], ct_sitk.GetSize()[0]
                valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < rows) & \
                        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < cols)
                if not np.any(valid):
                    print(f"Contour for '{structure_name}' on slice {slice_idx} is out of bounds. Skipping.")
                    continue
                pixel_coords = pixel_coords[valid]
                structures.append({
                    'pixel_coords': pixel_coords,
                    'structure_name': structure_name
                })
        return structures

    def add_controls(self, num_slices):
        self.current_slice = int(num_slices // 2)
        # Define button styles
        on_color = '#4CAF50'
        off_color = '#FF0000'
        button_width = 0.12
        button_height = 0.05
        button_spacing = 0.01

        total_buttons = 6  # CT, MR, Dose, Structures, Bias, Cropping
        total_button_width = total_buttons * button_width + (total_buttons - 1) * button_spacing
        start_x = (1 - total_button_width) / 2
        y_position = 0.05

        ax_ct = self.fig.add_axes([start_x, y_position, button_width, button_height])
        self.ct_button = Button(ax_ct, 'CT', color=on_color if self.ct_visible else off_color)
        self.ct_button.on_clicked(self.toggle_ct)

        ax_mr = self.fig.add_axes([start_x + (button_width + button_spacing), y_position, button_width, button_height])
        self.mr_button = Button(ax_mr, 'MR', color=on_color if self.mr_visible else off_color)
        self.mr_button.on_clicked(self.toggle_mr)

        ax_dose = self.fig.add_axes([start_x + 2*(button_width+button_spacing), y_position, button_width, button_height])
        self.dose_button = Button(ax_dose, 'Dose', color=on_color if self.dose_visible else off_color)
        self.dose_button.on_clicked(self.toggle_dose)

        ax_struct = self.fig.add_axes([start_x + 3*(button_width+button_spacing), y_position, button_width, button_height])
        self.struct_button = Button(ax_struct, 'Structures', color=on_color if self.structures_visible else off_color)
        self.struct_button.on_clicked(self.toggle_structures)

        ax_bias = self.fig.add_axes([start_x + 4*(button_width+button_spacing), y_position, button_width, button_height])
        self.bias_button = Button(ax_bias, 'Bias Corr', color=on_color if self.bias_correction_applied else off_color)
        self.bias_button.on_clicked(self.toggle_bias_correction)

        ax_crop = self.fig.add_axes([start_x + 5*(button_width+button_spacing), y_position, button_width, button_height])
        self.crop_button = Button(ax_crop, 'Cropping', color=on_color if self.cropping_applied else off_color)
        self.crop_button.on_clicked(self.toggle_cropping)

        # Slider for navigating slices
        ax_slider = self.fig.add_axes([0.05, 0.15, 0.9, 0.03])
        self.slice_slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=self.current_slice, valfmt='%d')
        self.slice_slider.on_changed(self.update_slice)

    def toggle_ct(self, event):
        self.ct_visible = not self.ct_visible
        self.ct_image_obj.set_visible(self.ct_visible)
        self.ct_button.ax.set_facecolor('#4CAF50' if self.ct_visible else '#FF0000')
        self.fig.canvas.draw()

    def toggle_mr(self, event):
        self.mr_visible = not self.mr_visible
        self.mr_image_obj.set_visible(self.mr_visible)
        self.mr_button.ax.set_facecolor('#4CAF50' if self.mr_visible else '#FF0000')
        self.fig.canvas.draw()

    def toggle_dose(self, event):
        self.dose_visible = not self.dose_visible
        self.dose_image_obj.set_visible(self.dose_visible)
        self.dose_button.ax.set_facecolor('#4CAF50' if self.dose_visible else '#FF0000')
        self.fig.canvas.draw()

    def toggle_structures(self, event):
        self.structures_visible = not self.structures_visible
        for line in self.structure_lines:
            line.set_visible(self.structures_visible)
        if self.legend_obj:
            self.legend_obj.set_visible(self.structures_visible)
        self.struct_button.ax.set_facecolor('#4CAF50' if self.structures_visible else '#FF0000')
        self.fig.canvas.draw()

    def toggle_bias_correction(self, event):
        self.bias_correction_applied = not self.bias_correction_applied
        self.bias_button.ax.set_facecolor('#4CAF50' if self.bias_correction_applied else '#FF0000')
        mr_array = self.mr_processed_array if self.bias_correction_applied else self.mr_original_array
        self.mr_image_obj.set_data(mr_array[self.current_slice, :, :])
        self.fig.canvas.draw()

    def toggle_cropping(self, event):
        self.cropping_applied = not self.cropping_applied
        self.crop_button.ax.set_facecolor('#4CAF50' if self.cropping_applied else '#FF0000')
        ct_array = self.ct_processed_array if self.cropping_applied else self.ct_original_array
        dose_array = self.dose_processed_array if self.cropping_applied else self.dose_original_array
        self.ct_image_obj.set_data(ct_array[self.current_slice, :, :])
        self.dose_image_obj.set_data(dose_array[self.current_slice, :, :])
        if np.max(dose_array) > 0:
            self.dose_image_obj.set_clim(vmin=0, vmax=np.percentile(dose_array, 99))
        else:
            self.dose_image_obj.set_clim(vmin=0, vmax=0)
        self.update_structures()
        self.fig.canvas.draw()

    def update_structures(self):
        for line in self.structure_lines:
            line.remove()
        self.structure_lines.clear()
        structures_on_slice = self.get_structures_on_slice(self.current_slice)
        for structure in structures_on_slice:
            contour_points = structure['pixel_coords']
            structure_name = structure['structure_name']
            if contour_points.size == 0:
                continue
            color = self.structure_colors.get(structure_name, 'white')
            line, = self.ax.plot(contour_points[:, 1], contour_points[:, 0],
                                 linewidth=2, label=structure_name, color=color)
            self.structure_lines.append(line)
        if self.legend_obj:
            self.legend_obj.remove()
            self.legend_obj = None
        if self.structure_lines and self.structures_visible:
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.legend_obj = self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        self.fig.canvas.draw()

    def update_slice(self, val):
        self.current_slice = int(val)
        ct_array = self.ct_processed_array if self.cropping_applied else self.ct_original_array
        mr_array = self.mr_processed_array if self.bias_correction_applied else self.mr_original_array
        dose_array = self.dose_processed_array if self.cropping_applied else self.dose_original_array
        self.ct_image_obj.set_data(ct_array[self.current_slice, :, :])
        self.mr_image_obj.set_data(mr_array[self.current_slice, :, :])
        self.dose_image_obj.set_data(dose_array[self.current_slice, :, :])
        if np.max(dose_array) > 0:
            self.dose_image_obj.set_clim(vmin=0, vmax=np.percentile(dose_array, 99))
        else:
            self.dose_image_obj.set_clim(vmin=0, vmax=0)
        self.update_structures()
        for text_obj in self.annotation_texts:
            text_obj.remove()
        self.annotation_texts.clear()
        if self.rois_without_contours:
            text_y = 0.95
            for roi_name in self.rois_without_contours:
                text_obj = self.ax.text(0.05, text_y, f"No contour data for ROI: {roi_name}",
                                        transform=self.ax.transAxes, fontsize=10, color='red')
                self.annotation_texts.append(text_obj)
                text_y -= 0.03
        self.ax.set_title(f"Patient: {self.patient_id}, {self.mr_modality} - Slice {self.current_slice}", fontsize=16)
        self.fig.canvas.draw()

# =============================================================================
# GBM Processor Class
# =============================================================================
class GBMProcessor:
    """
    Coordinates the loading, processing, and visualization of GBM patient imaging data.
    """
    def __init__(self, patient_id, mr_modality="T1CE"):
        self.patient_id = patient_id
        self.mr_modality = mr_modality
        self.dicom_loader = DICOMLoader(patient_id)
        self.image_processor = ImageProcessor()
        self.rtstruct_processor_original = None
        self.rtstruct_processor_cropped = None
        self.viewer = None

    def run(self):
        self.dicom_loader.load_studies()
        self.dicom_loader.load_series()
        self.dicom_loader.validate_series()

        print("Loading CT DICOM series...")
        ct_original = self.dicom_loader.load_dicom_series(self.dicom_loader.series_dict['CT'])
        if ct_original is None:
            raise FileNotFoundError(f"Could not load CT series from {self.dicom_loader.series_dict['CT']}")

        mr_series_key = f"MR_{self.mr_modality}"
        if mr_series_key not in self.dicom_loader.series_dict:
            raise FileNotFoundError(f"No MR series found for modality {self.mr_modality}.")
        print(f"Loading MR DICOM series for {self.mr_modality}...")
        mr_original = self.dicom_loader.load_dicom_series(self.dicom_loader.series_dict[mr_series_key])
        if mr_original is None:
            raise FileNotFoundError(f"Could not load MR series from {self.dicom_loader.series_dict[mr_series_key]}")

        mr_original = sitk.Cast(mr_original, sitk.sitkFloat32)
        print("Performing N4 Bias Field Correction on MR image...")
        mr_processed = self.image_processor.bias_correction(mr_original)

        print("Registering MR to CT...")
        registered_mr, _ = self.image_processor.register_images(ct_original, mr_processed)
        print("MR to CT registration completed.")

        print("Loading RTDOSE...")
        dose_original, rtdose_dataset = self.dicom_loader.load_rtdose()
        if dose_original is None:
            print("Failed to load RTDOSE data. Creating empty dose image.")
            dose_original = sitk.Image(ct_original.GetSize(), sitk.sitkFloat32)
            dose_original.CopyInformation(ct_original)
        else:
            print("RTDOSE loaded.")
            same_origin = np.allclose(dose_original.GetOrigin(), ct_original.GetOrigin())
            same_spacing = np.allclose(dose_original.GetSpacing(), ct_original.GetSpacing())
            same_direction = np.allclose(dose_original.GetDirection(), ct_original.GetDirection())
            need_registration = not (same_origin and same_spacing and same_direction)
            if need_registration:
                print("Dose and CT are not aligned. Registering Dose to CT...")
                registered_dose, _ = self.image_processor.register_images(ct_original, dose_original)
                if registered_dose is None:
                    print("Dose registration failed. Creating empty dose image.")
                    registered_dose = sitk.Image(ct_original.GetSize(), sitk.sitkFloat32)
                    registered_dose.CopyInformation(ct_original)
                else:
                    print("Dose to CT registration completed.")
                    dose_original = registered_dose
            else:
                print("Dose is already aligned with CT. No registration needed.")
            print("Resampling RTDOSE to CT grid...")
            resampled_dose = self.image_processor.resample_image(ct_original, dose_original)
            if resampled_dose is None:
                print("Resampling failed. Creating empty dose image.")
                resampled_dose = sitk.Image(ct_original.GetSize(), sitk.sitkFloat32)
                resampled_dose.CopyInformation(ct_original)
            else:
                print("Dose resampled to CT grid.")
                dose_original = resampled_dose

        print("Creating brain mask...")
        brain_mask = self.image_processor.create_brain_mask(registered_mr)
        print("Applying brain mask to images...")
        masked_ct, masked_mr, masked_dose = self.image_processor.apply_brain_mask(ct_original, registered_mr, dose_original, brain_mask)
        print("Cropping images to brain bounding box...")
        cropped_ct, cropped_mr, cropped_dose, bounding_box = self.crop_images(ct_original, registered_mr, dose_original, brain_mask)
        print("Cropping completed.")

        ct_original_array = sitk.GetArrayFromImage(ct_original)
        ct_processed_array = sitk.GetArrayFromImage(cropped_ct)
        mr_original_array = sitk.GetArrayFromImage(mr_original)
        mr_processed_array = sitk.GetArrayFromImage(cropped_mr)
        dose_original_array = sitk.GetArrayFromImage(dose_original)
        dose_processed_array = sitk.GetArrayFromImage(cropped_dose)

        print("Calculating Z positions for original CT slices...")
        ct_origin = ct_original.GetOrigin()
        ct_spacing = ct_original.GetSpacing()
        ct_size = ct_original.GetSize()
        ct_slice_z_positions_original = [ct_origin[2] + idx * ct_spacing[2] for idx in range(ct_size[2])]
        print("Original CT slice Z positions calculated.")

        print("Calculating Z positions for cropped CT slices...")
        ct_origin_cropped = cropped_ct.GetOrigin()
        ct_spacing_cropped = cropped_ct.GetSpacing()
        ct_size_cropped = cropped_ct.GetSize()
        ct_slice_z_positions_cropped = [ct_origin_cropped[2] + idx * ct_spacing_cropped[2] for idx in range(ct_size_cropped[2])]
        print("Cropped CT slice Z positions calculated.")

        print("Processing RTSTRUCT for original images...")
        rtstruct_datasets = [data for (_, data) in self.dicom_loader.rtstruct_files]
        self.rtstruct_processor_original = RTStructProcessor(rtstruct_datasets)
        self.rtstruct_processor_original.map_contours_to_slices(ct_slice_z_positions_original)
        slice_contours_original = self.rtstruct_processor_original.slice_contours
        rois_without_contours_original = self.rtstruct_processor_original.rois_without_contours
        all_structure_names_original = self.rtstruct_processor_original.all_structure_names
        print("RTSTRUCT processing for original images completed.")

        print("Processing RTSTRUCT for cropped images...")
        self.rtstruct_processor_cropped = RTStructProcessor(rtstruct_datasets)
        self.rtstruct_processor_cropped.map_contours_to_slices(ct_slice_z_positions_cropped)
        slice_contours_cropped = self.rtstruct_processor_cropped.slice_contours
        rois_without_contours_cropped = self.rtstruct_processor_cropped.rois_without_contours
        all_structure_names_cropped = self.rtstruct_processor_cropped.all_structure_names
        print("RTSTRUCT processing for cropped images completed.")

        rois_without_contours = rois_without_contours_original.union(rois_without_contours_cropped)
        all_structure_names = all_structure_names_original.union(all_structure_names_cropped)

        self.viewer = InteractiveViewer(
            ct_original_array=ct_original_array,
            ct_processed_array=ct_processed_array,
            dose_original_array=dose_original_array,
            dose_processed_array=dose_processed_array,
            mr_original_array=mr_original_array,
            mr_processed_array=mr_processed_array,
            slice_contours_original=slice_contours_original,
            slice_contours_cropped=slice_contours_cropped,
            rois_without_contours=rois_without_contours,
            ct_original_sitk=ct_original,
            ct_processed_sitk=cropped_ct,
            dose_original_sitk=dose_original,
            dose_processed_sitk=cropped_dose,
            mr_original_sitk=mr_original,
            mr_processed_sitk=cropped_mr,
            bounding_box=bounding_box,
            apply_cropping=True,
            apply_bias_correction=True,
            all_structure_names=all_structure_names,
            patient_id=self.patient_id,
            mr_modality=self.mr_modality
        )
        # Pass the contour mappings to the viewer (for use in slice display)
        self.viewer.slice_contours_original = slice_contours_original
        self.viewer.slice_contours_cropped = slice_contours_cropped

        fig = self.viewer.launch()
        return fig

    def crop_images(self, ct_image_sitk, mr_image_sitk, dose_image_sitk, brain_mask):
        # Create brain mask and compute bounding box from connected components
        brain_mask = self.image_processor.create_brain_mask(mr_image_sitk)
        connected_component_filter = sitk.ConnectedComponentImageFilter()
        brain_cc = connected_component_filter.Execute(brain_mask)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(brain_cc)
        largest_label = None
        largest_size = 0
        for label in label_shape_filter.GetLabels():
            size = label_shape_filter.GetPhysicalSize(label)
            if size > largest_size:
                largest_size = size
                largest_label = label
        brain_mask = sitk.BinaryThreshold(brain_cc,
                                          lowerThreshold=largest_label,
                                          upperThreshold=largest_label,
                                          insideValue=1,
                                          outsideValue=0)
        # Get the original bounding box from the label
        bounding_box = label_shape_filter.GetBoundingBox(largest_label)  # (minX, minY, minZ, sizeX, sizeY, sizeZ)
        index = list(bounding_box[:3])
        size = list(bounding_box[3:])
        
        # Add margin (in mm) around the bounding box and convert to voxels
        margin_mm = 10
        spacing = np.array(mr_image_sitk.GetSpacing())
        margin_voxels = [int(margin_mm / sp) for sp in spacing]
        index = [max(0, idx - margin_voxels[i]) for i, idx in enumerate(index)]
        size = [min(mr_image_sitk.GetSize()[i] - index[i], size[i] + 2 * margin_voxels[i]) for i in range(3)]
        
        # ---- Adjust the crop region to be a square in the axial plane (first two dimensions) ----
        # Get image dimensions (assumed to be (width, height, depth))
        img_size = mr_image_sitk.GetSize()
        # Determine desired square side length as the maximum of the current width and height of the crop
        desired_square = max(size[0], size[1])
        # Ensure the square region does not exceed image boundaries
        if index[0] + desired_square > img_size[0]:
            desired_square = img_size[0] - index[0]
        if index[1] + desired_square > img_size[1]:
            desired_square = img_size[1] - index[1]
        # Re-center the crop in x and y based on the original bounding box center
        center_x = index[0] + size[0] // 2
        center_y = index[1] + size[1] // 2
        new_index_x = max(0, center_x - desired_square // 2)
        new_index_y = max(0, center_y - desired_square // 2)
        # Make sure the square fits within the image boundaries
        if new_index_x + desired_square > img_size[0]:
            new_index_x = img_size[0] - desired_square
        if new_index_y + desired_square > img_size[1]:
            new_index_y = img_size[1] - desired_square
        index[0] = new_index_x
        index[1] = new_index_y
        size[0] = desired_square
        size[1] = desired_square
        # ---------------------------------------------------------------------------------------------

        cropped_ct = sitk.RegionOfInterest(ct_image_sitk, size=size, index=index)
        cropped_mr = sitk.RegionOfInterest(mr_image_sitk, size=size, index=index)
        cropped_dose = sitk.RegionOfInterest(dose_image_sitk, size=size, index=index)
        cropped_ct = self.image_processor.adjust_origin_after_cropping(ct_image_sitk, cropped_ct, index)
        cropped_mr = self.image_processor.adjust_origin_after_cropping(mr_image_sitk, cropped_mr, index)
        cropped_dose = self.image_processor.adjust_origin_after_cropping(dose_image_sitk, cropped_dose, index)
        return cropped_ct, cropped_mr, cropped_dose, bounding_box

# =============================================================================
# Tkinter Application with Dropdown Menus
# =============================================================================

class GBMViewerApp(tk.Tk):
    """
    Tkinter-based application that allows users to select a patient and MR modality,
    then loads and displays the interactive viewer for image visualization and processing control.
    """
    def __init__(self, patients, modalities):
        super().__init__()
        self.title("GBM Viewer")
        self.geometry("1400x900")

        # Variables for selected patient and modality
        self.selected_patient = tk.IntVar(value=patients[0])
        self.selected_modality = tk.StringVar(value=modalities[0])

        # Top frame for dropdown menus
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Patient:").pack(side=tk.LEFT, padx=(0,5))
        patient_menu = ttk.Combobox(control_frame, textvariable=self.selected_patient,
                                    values=patients, state="readonly", width=5)
        patient_menu.pack(side=tk.LEFT, padx=(0,20))

        tk.Label(control_frame, text="MR Modality:").pack(side=tk.LEFT, padx=(0,5))
        modality_menu = ttk.Combobox(control_frame, textvariable=self.selected_modality,
                                     values=modalities, state="readonly", width=10)
        modality_menu.pack(side=tk.LEFT, padx=(0,20))

        load_button = tk.Button(control_frame, text="Load Viewer", command=self.load_viewer)
        load_button.pack(side=tk.LEFT, padx=10)

        # Frame for matplotlib canvas
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def load_viewer(self):
        # Clear existing canvas (if any)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        patient_id = self.selected_patient.get()
        modality = self.selected_modality.get()
        print(f"Loading viewer for Patient: {patient_id}, Modality: {modality}")
        processor = GBMProcessor(patient_id, mr_modality=modality)
        fig = processor.run()  # This returns the matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# =============================================================================
# Main
# =============================================================================
def main():
    patients = list(range(1, 181))   # Patient IDs 1 to 180
    modalities = ["T1CE", "T2FLAIR"]

    app = GBMViewerApp(patients, modalities)
    app.mainloop()

if __name__ == "__main__":
    main()
