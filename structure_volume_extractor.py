"""
Module for extracting structure volumes from RTSTRUCT DICOM data.
This class loads CT images and RTSTRUCT contours for a given patient,
generates binary masks for each structure, and computes the volume (in cc).
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon


class StructureVolumeExtractor:
    """
    Class to extract and compute structure volumes from DICOM data.
    """

    def __init__(self, patient_id: int, base_path: str = "Burdenko-GBM-Progression", verbose: bool = True) -> None:
        """
        Initialize the extractor with a patient ID and a base path.
        
        Args:
            patient_id (int): Patient identifier.
            base_path (str): Root directory where patient data is stored.
            verbose (bool): If True, log detailed messages.
        """
        self.patient_id: int = patient_id
        self.base_path: str = base_path
        self.verbose: bool = verbose

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        self.patient_file_name: str = f"Burdenko-GBM-{patient_id:03d}"
        self.patient_path: str = os.path.join(base_path, self.patient_file_name)
        
        self.ct_series_path: Optional[str] = None
        self.rtstruct_files: List[Tuple[str, pydicom.dataset.FileDataset]] = []
        self.ct_image_sitk: Optional[sitk.Image] = None
        self.ct_array: Optional[np.ndarray] = None
        self.structure_masks: Dict[str, np.ndarray] = {}
        self.all_structure_names: set = set()

        self.load_metadata()

    def log(self, message: str) -> None:
        """Log a message if verbosity is enabled."""
        if self.verbose:
            self.logger.info(message)

    def load_metadata(self) -> None:
        """
        Load metadata from the patient directory.
        This method finds the radiotherapy study, identifies the CT series and collects RTSTRUCT files.
        """
        if not os.path.isdir(self.patient_path):
            raise FileNotFoundError(f"Patient path not found: {self.patient_path}")

        studies = [d for d in os.listdir(self.patient_path)
                   if os.path.isdir(os.path.join(self.patient_path, d))]
        radiotherapy_study = next((s for s in studies if "Radiotherapy planning" in s), None)
        if not radiotherapy_study:
            raise FileNotFoundError("No radiotherapy study found for patient path.")

        rt_study_path = os.path.join(self.patient_path, radiotherapy_study)
        series = [d for d in os.listdir(rt_study_path)
                  if os.path.isdir(os.path.join(rt_study_path, d))]
        self.log(f"[Patient {self.patient_id}] Series found in radiotherapy study:")
        for s in series:
            self.log(f"  - {s}")
            series_path = os.path.join(rt_study_path, s)
            dicom_files = self.find_dicom_files(series_path)
            if not dicom_files:
                self.log(f"    No valid DICOM files in series: {s}")
                continue
            first_dicom = dicom_files[0][1]
            modality = getattr(first_dicom, 'Modality', '')
            series_description = getattr(first_dicom, 'SeriesDescription', '')

            if modality == "CT":
                self.ct_series_path = series_path
                self.log(f"    CT series identified: {s}")
            elif modality == "RTSTRUCT":
                self.rtstruct_files.extend(dicom_files)
                self.log(f"    RTSTRUCT files added: {s}")

        if not self.ct_series_path:
            raise FileNotFoundError("No CT series found for the patient.")
        if not self.rtstruct_files:
            raise FileNotFoundError("No RTSTRUCT files found for the patient.")

    def find_dicom_files(self, series_path: str) -> List[Tuple[str, pydicom.dataset.FileDataset]]:
        """
        Find valid DICOM files in a series directory.
        
        Args:
            series_path (str): Directory path to search.
        
        Returns:
            List of tuples (filepath, dicom_data) for valid DICOM files.
        """
        files = os.listdir(series_path)

        def process_file(file: str) -> Optional[Tuple[str, pydicom.dataset.FileDataset]]:
            if file.startswith('._') or not file.endswith(".dcm"):
                return None
            filepath = os.path.join(series_path, file)
            try:
                dicom_data = pydicom.dcmread(filepath, stop_before_pixels=True)
                return (filepath, dicom_data)
            except (pydicom.errors.InvalidDicomError, AttributeError):
                self.log(f"    Skipping invalid or incomplete DICOM file: {filepath}")
                return None

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_file, files)

        return [res for res in results if res is not None]

    def load_ct_image(self) -> None:
        """
        Load the CT series using SimpleITK.
        Updates self.ct_image_sitk and self.ct_array.
        """
        if self.ct_series_path is None:
            raise FileNotFoundError("CT series path is not set.")
        self.log(f"[Patient {self.patient_id}] Loading CT series from {self.ct_series_path}")
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(self.ct_series_path)
        if not series_IDs:
            raise FileNotFoundError(f"No DICOM series found in directory {self.ct_series_path}")
        series_file_names = reader.GetGDCMSeriesFileNames(self.ct_series_path, series_IDs[0])
        reader.SetFileNames(series_file_names)
        try:
            self.ct_image_sitk = reader.Execute()
            self.ct_array = sitk.GetArrayFromImage(self.ct_image_sitk)
            self.log(f"[Patient {self.patient_id}] CT image loaded with shape: {self.ct_array.shape}")
        except Exception as e:
            raise RuntimeError(f"Error loading CT series: {e}")

    def load_rtstruct(self) -> None:
        """
        Process all RTSTRUCT files concurrently.
        """
        self.log(f"[Patient {self.patient_id}] Loading RTSTRUCT files...")
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_rtstruct_file, self.rtstruct_files)

    def process_rtstruct_file(self, rtstruct_file: Tuple[str, pydicom.dataset.FileDataset]) -> None:
        """
        Process a single RTSTRUCT file to generate masks for each structure.
        
        Args:
            rtstruct_file (Tuple[str, pydicom.dataset.FileDataset]): Tuple of filepath and DICOM data.
        """
        filepath, rtstruct_data = rtstruct_file
        self.log(f"  Processing RTSTRUCT file: {filepath}")
        if not hasattr(rtstruct_data, 'StructureSetROISequence'):
            self.log("    RTSTRUCT data missing 'StructureSetROISequence'. Skipping.")
            return

        # Build ROI dictionary and initialize structure masks
        roi_dict: Dict[int, str] = {}
        for roi in rtstruct_data.StructureSetROISequence:
            if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                roi_name = roi.ROIName.strip()
                roi_dict[roi.ROINumber] = roi_name
                self.all_structure_names.add(roi_name)
                if roi_name not in self.structure_masks and self.ct_array is not None:
                    self.structure_masks[roi_name] = np.zeros_like(self.ct_array, dtype=np.uint8)
            else:
                self.log("    ROI in StructureSetROISequence missing 'ROINumber' or 'ROIName'. Skipping.")
                continue

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
                            self.log(f"    Invalid ContourData format for structure '{roi_name}'")
                            continue
                        pixel_coords = self.transform_physical_to_pixel(contour_points)
                        if pixel_coords is not None and pixel_coords.shape[0] >= 3:
                            z_position = contour_points[0, 2]
                            slice_idx = self.get_slice_index(z_position)
                            if slice_idx is not None and 0 <= slice_idx < self.ct_array.shape[0]:
                                rr, cc = polygon(pixel_coords[:, 1], pixel_coords[:, 0],
                                                  shape=self.ct_array.shape[1:])
                                self.structure_masks[roi_name][slice_idx, rr, cc] = 1
                            else:
                                self.log(f"    Calculated slice index {slice_idx} out of CT image bounds. Skipping.")
        else:
            self.log("    RTSTRUCT data missing 'ROIContourSequence'. Skipping.")

    def transform_physical_to_pixel(self, contour_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform physical (world) coordinates to pixel indices in the CT image.
        
        Args:
            contour_points (np.ndarray): Nx3 array of physical coordinates.
        
        Returns:
            np.ndarray: Nx2 array of pixel coordinates (col, row), or None if an error occurs.
        """
        try:
            pixel_coords = []
            for point in contour_points:
                index = self.ct_image_sitk.TransformPhysicalPointToIndex(tuple(point))
                col, row = index[0], index[1]
                if (0 <= col < self.ct_image_sitk.GetSize()[0]) and (0 <= row < self.ct_image_sitk.GetSize()[1]):
                    pixel_coords.append([col, row])
                else:
                    self.log(f"    Physical point {point} is out of CT image bounds. Skipping contour.")
                    return None
            return np.array(pixel_coords)
        except Exception as e:
            self.log(f"    Error transforming physical points to pixel coordinates: {e}")
            return None

    def get_slice_index(self, z_position: float) -> Optional[int]:
        """
        Calculate the CT slice index corresponding to a given z-position.
        
        Args:
            z_position (float): Z coordinate in physical space.
        
        Returns:
            int: Slice index, or None if CT image is not loaded.
        """
        if self.ct_image_sitk is None:
            self.log("    CT image not loaded. Cannot determine slice index.")
            return None
        origin = self.ct_image_sitk.GetOrigin()
        spacing = self.ct_image_sitk.GetSpacing()
        z_origin = origin[2]
        slice_idx = round((z_position - z_origin) / spacing[2])
        if 0 <= slice_idx < self.ct_array.shape[0]:
            return slice_idx
        return None

    def calculate_volumes(self) -> pd.DataFrame:
        """
        Calculate the volumes (in cc) for each structure based on the generated masks.
        
        Returns:
            pd.DataFrame: DataFrame containing structure names and their volumes in cc.
        """
        self.log(f"[Patient {self.patient_id}] Calculating structure volumes...")
        if self.ct_image_sitk is None:
            raise ValueError("CT image not loaded. Cannot calculate volumes.")

        spacing = self.ct_image_sitk.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_cc = voxel_volume_mm3 / 1000.0

        volumes = []
        for structure, mask in self.structure_masks.items():
            voxel_count = np.sum(mask)
            volume_cc = voxel_count * voxel_volume_cc
            volumes.append({'Structure': structure, 'Volume_cc': volume_cc})
            self.log(f"  Structure '{structure}': {volume_cc:.2f} cc")

        return pd.DataFrame(volumes)

    def get_structure_volumes(self) -> pd.DataFrame:
        """
        Execute the full workflow: load CT image, process RTSTRUCT files, and compute volumes.
        
        Returns:
            pd.DataFrame: DataFrame with the computed volumes.
        """
        self.load_ct_image()
        self.load_rtstruct()
        return self.calculate_volumes()


if __name__ == '__main__':
    extractor = StructureVolumeExtractor(patient_id=97, verbose=True)
    volumes_df = extractor.get_structure_volumes()
    print(volumes_df)
