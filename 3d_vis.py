"""
Script for processing DICOM data for radiotherapy planning.
Processes CT, MR, RTSTRUCT, RTPLAN, and RTDOSE data, performs image registration,
downsampling, and creates a 3D visualization using Plotly.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pydicom
import plotly.graph_objects as go
import plotly.express as px
import SimpleITK as sitk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# Global Parameters and Patient Directory Setup
# =============================================================================

PATIENT_ID = 97 # patient to plot
PATIENT_FILE_NAME = f"Burdenko-GBM-{PATIENT_ID:03d}"
PATIENT_PATH = os.path.join("Burdenko-GBM-Progression", PATIENT_FILE_NAME)


# =============================================================================
# Helper Functions
# =============================================================================

def find_dicom_files(series_path: str) -> List[Tuple[str, pydicom.dataset.FileDataset]]:
    """
    Find and return all valid DICOM files in a given directory.

    Parameters:
        series_path (str): Directory path to search for DICOM files.

    Returns:
        List of tuples with (filepath, dicom_data).
    """
    dicom_files = []
    for file in os.listdir(series_path):
        if file.startswith('._') or not file.endswith(".dcm"):
            continue
        filepath = os.path.join(series_path, file)
        try:
            dicom_data = pydicom.dcmread(filepath, force=True)
            dicom_files.append((filepath, dicom_data))
        except pydicom.errors.InvalidDicomError:
            logging.warning("Invalid DICOM file skipped: %s", filepath)
        except AttributeError:
            logging.warning("File missing required attributes: %s", filepath)
    return dicom_files


def load_dicom_series(series_path: str) -> Optional[sitk.Image]:
    """
    Loads a DICOM series from the specified directory using SimpleITK.

    Parameters:
        series_path (str): Directory containing the DICOM series.

    Returns:
        SimpleITK image if successful; otherwise, None.
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(series_path)
    if not series_IDs:
        logging.error("No DICOM series found in directory: %s", series_path)
        return None
    series_file_names = reader.GetGDCMSeriesFileNames(series_path, series_IDs[0])
    reader.SetFileNames(series_file_names)
    return reader.Execute()


def physical_to_pixel_coords(contour_points: np.ndarray,
                             dicom_dataset: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Converts physical (world) coordinates to pixel coordinates for a DICOM image.

    Parameters:
        contour_points (np.ndarray): Nx3 array of (x, y, z) points.
        dicom_dataset (pydicom.dataset.FileDataset): DICOM dataset of the image slice.

    Returns:
        Nx2 numpy array of (row, column) indices.
    """
    image_position = np.array(dicom_dataset.ImagePositionPatient)
    image_orientation = np.array(dicom_dataset.ImageOrientationPatient)
    pixel_spacing = np.array(dicom_dataset.PixelSpacing, dtype=np.float64)
    row_direction = image_orientation[3:6]
    col_direction = image_orientation[0:3]
    delta = contour_points - image_position
    indices = np.zeros((delta.shape[0], 2))
    indices[:, 0] = np.dot(delta, row_direction) / pixel_spacing[1]
    indices[:, 1] = np.dot(delta, col_direction) / pixel_spacing[0]
    return indices


def register_mr_to_ct(ct_image: sitk.Image, mr_image: sitk.Image) -> Tuple[sitk.Image, sitk.Transform]:
    """
    Registers an MR image to a CT image using SimpleITK.

    Parameters:
        ct_image (sitk.Image): CT volume.
        mr_image (sitk.Image): MR volume.

    Returns:
        Tuple containing the registered MR image and the transformation.
    """
    ct_image = sitk.Cast(ct_image, sitk.sitkFloat32)
    mr_image = sitk.Cast(mr_image, sitk.sitkFloat32)
    
    initial_transform = sitk.CenteredTransformInitializer(
        ct_image,
        mr_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-6,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(ct_image, mr_image)
    registered_mr = sitk.Resample(
        mr_image,
        ct_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        mr_image.GetPixelID()
    )
    
    return registered_mr, final_transform


def downsample_volume(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsamples a 3D volume by a specified factor.

    Parameters:
        volume (np.ndarray): 3D volume array.
        factor (int): Downsampling factor.

    Returns:
        Downsampled 3D numpy array.
    """
    return volume[::factor, ::factor, ::factor]


def get_beam_geometries(rtplan: pydicom.dataset.FileDataset) -> List[Dict[str, Any]]:
    """
    Extracts beam geometries from RTPLAN data.

    Parameters:
        rtplan (pydicom.dataset.FileDataset): RTPLAN dataset.

    Returns:
        List of dictionaries with beam geometry information.
    """
    beam_geometries = []
    for beam in rtplan.BeamSequence:
        beam_info = {
            'BeamName': beam.get('BeamName', f"Beam {beam.BeamNumber}"),
            'GantryAngle': beam.ControlPointSequence[0].GantryAngle,
            'CollimatorAngle': beam.ControlPointSequence[0].BeamLimitingDeviceAngle,
            'IsocenterPosition': beam.ControlPointSequence[0].IsocenterPosition,
            'SourceAxisDistance': beam.SourceAxisDistance,
            'LeafPositions': []
        }
        if 'BeamLimitingDeviceSequence' in beam:
            for device in beam.BeamLimitingDeviceSequence:
                if device.RTBeamLimitingDeviceType in ['MLCX', 'MLCY']:
                    leaf_positions = []
                    for cp in beam.ControlPointSequence:
                        if 'BeamLimitingDevicePositionSequence' in cp:
                            for bldp in cp.BeamLimitingDevicePositionSequence:
                                if bldp.RTBeamLimitingDeviceType == device.RTBeamLimitingDeviceType:
                                    leaf_positions.append(bldp.LeafJawPositions)
                    beam_info['LeafPositions'].append({
                        'DeviceType': device.RTBeamLimitingDeviceType,
                        'Positions': leaf_positions
                    })
        beam_geometries.append(beam_info)
    return beam_geometries


def load_rtdose_data(rtdose_files: List[Tuple[str, pydicom.dataset.FileDataset]],
                     rtdose_series_paths: List[str],
                     ct_image: sitk.Image) -> Tuple[Optional[sitk.Image], Optional[pydicom.dataset.FileDataset]]:
    """
    Loads RTDOSE data from file or series.

    Parameters:
        rtdose_files (list): List of single RTDOSE file tuples.
        rtdose_series_paths (list): List of RTDOSE series directories.
        ct_image (sitk.Image): CT image (for reference).

    Returns:
        Tuple of (dose image as sitk.Image, dose metadata dataset).
    """
    if rtdose_files:
        rtdose_file_path, rtdose_dataset = rtdose_files[0]
        try:
            dose_image = sitk.ReadImage(rtdose_file_path)
            return dose_image, rtdose_dataset
        except Exception as e:
            logging.error("Error loading RTDOSE file %s: %s", rtdose_file_path, e)
            return None, None
    elif rtdose_series_paths:
        series_path = rtdose_series_paths[0]
        dose_image = load_dicom_series(series_path)
        if dose_image is None:
            logging.error("Could not load RTDOSE series from %s", series_path)
            return None, None
        dose_files = find_dicom_files(series_path)
        if not dose_files:
            logging.error("No DICOM files found in RTDOSE series.")
            return dose_image, None
        rtdose_dataset = dose_files[0][1]
        return dose_image, rtdose_dataset
    else:
        logging.error("No RTDOSE data available.")
        return None, None


def create_3d_visualization(ct_array: np.ndarray,
                            mr_array: np.ndarray,
                            dose_array: np.ndarray,
                            slice_contours: Dict[int, List[Dict[str, Any]]],
                            beam_geometries: List[Dict[str, Any]],
                            all_structure_names: Set[str],
                            rois_without_contours: List[str]) -> go.Figure:
    """
    Creates a 3D visualization of CT, MR, Dose, RTSTRUCT contours, and RTPLAN beam geometries.

    Parameters:
        ct_array (np.ndarray): Downsampled CT volume.
        mr_array (np.ndarray): Downsampled registered MR volume.
        dose_array (np.ndarray): Downsampled dose volume.
        slice_contours (dict): Mapping of CT slice indices to contour info.
        beam_geometries (list): Beam geometry information from RTPLAN.
        all_structure_names (set): Unique structure names.
        rois_without_contours (list): List of ROIs missing contour data.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # Define colors for structures
    unique_structures = sorted(list(all_structure_names))
    cmap = px.colors.qualitative.Plotly
    structure_colors = {name: cmap[i % len(cmap)] for i, name in enumerate(unique_structures)}
    
    # CT isosurface
    fig.add_trace(go.Isosurface(
        x=np.linspace(0, ct_array.shape[2], ct_array.shape[2]),
        y=np.linspace(0, ct_array.shape[1], ct_array.shape[1]),
        z=np.linspace(0, ct_array.shape[0], ct_array.shape[0]),
        value=ct_array.flatten(),
        isomin=np.percentile(ct_array, 10),
        isomax=np.percentile(ct_array, 90),
        surface_count=2,
        colorscale='Gray',
        showscale=False,
        opacity=0.1,
        name='CT'
    ))
    
    # MR isosurface
    fig.add_trace(go.Isosurface(
        x=np.linspace(0, mr_array.shape[2], mr_array.shape[2]),
        y=np.linspace(0, mr_array.shape[1], mr_array.shape[1]),
        z=np.linspace(0, mr_array.shape[0], mr_array.shape[0]),
        value=mr_array.flatten(),
        isomin=np.percentile(mr_array, 10),
        isomax=np.percentile(mr_array, 90),
        surface_count=2,
        colorscale='Blues',
        showscale=False,
        opacity=0.1,
        name='MR'
    ))
    
    # Dose isosurface (if dose values exist)
    if np.max(dose_array) > 0:
        fig.add_trace(go.Isosurface(
            x=np.linspace(0, dose_array.shape[2], dose_array.shape[2]),
            y=np.linspace(0, dose_array.shape[1], dose_array.shape[1]),
            z=np.linspace(0, dose_array.shape[0], dose_array.shape[0]),
            value=dose_array.flatten(),
            isomin=np.percentile(dose_array, 30),
            isomax=np.percentile(dose_array, 70),
            surface_count=2,
            colorscale='Jet',
            showscale=False,
            opacity=0.2,
            name='Dose'
        ))
    
    # RTSTRUCT contours as 3D lines
    for slice_idx, contours in slice_contours.items():
        for contour in contours:
            points = contour['points']
            structure_name = contour['structure_name']
            color = structure_colors.get(structure_name, 'white')
            # Downsample contour points to reduce memory usage
            contour_ds = points[::2]
            fig.add_trace(go.Scatter3d(
                x=contour_ds[:, 0],
                y=contour_ds[:, 1],
                z=contour_ds[:, 2],
                mode='lines',
                line=dict(color=color, width=2),
                name=structure_name,
                showlegend=False
            ))
    
    # RTPLAN beam geometries as beam lines
    for beam in beam_geometries:
        isocenter = beam['IsocenterPosition']
        gantry_angle = beam['GantryAngle']
        SAD = beam['SourceAxisDistance']
        angle_rad = np.deg2rad(gantry_angle)
        source_x = isocenter[0] + SAD * np.sin(angle_rad)
        source_y = isocenter[1]
        source_z = isocenter[2] - SAD * np.cos(angle_rad)
        fig.add_trace(go.Scatter3d(
            x=[source_x, isocenter[0]],
            y=[source_y, isocenter[1]],
            z=[source_z, isocenter[2]],
            mode='lines',
            line=dict(color='yellow', width=4),
            name=beam['BeamName'],
            hoverinfo='text',
            text=[beam['BeamName']]
        ))
    
    # Annotate ROIs without contours
    for roi_name in rois_without_contours:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=0),
            text=[f"No contour data for ROI: {roi_name}"],
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Visualization of CT, MR, Dose, RTSTRUCT, and RTPLAN",
        legend=dict(itemsizing='constant'),
        width=1000,
        height=800
    )
    
    return fig


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def main():
    # Find patient studies and select the Radiotherapy planning study.
    studies = [d for d in os.listdir(PATIENT_PATH) if os.path.isdir(os.path.join(PATIENT_PATH, d))]
    rt_study = next((study for study in studies if "Radiotherapy planning" in study), None)
    if rt_study is None:
        raise FileNotFoundError("No radiotherapy study found for this patient.")
    logging.info("Radiotherapy Study Found: %s", rt_study)
    
    rt_study_path = os.path.join(PATIENT_PATH, rt_study)
    series_dirs = [d for d in os.listdir(rt_study_path)
                   if os.path.isdir(os.path.join(rt_study_path, d)) and not d.startswith('._')]
    logging.info("Series found in Radiotherapy Study: %s", series_dirs)
    
    # Initialize variables for different modalities
    ct_series_path = None
    mr_t2_series_path = None
    rtstruct_files = []
    rtstruct_data_list = []
    rtplan_files = []
    rtdose_files = []
    rtdose_series_paths = []
    
    # Identify series by modality
    for s in series_dirs:
        series_path = os.path.join(rt_study_path, s)
        dicom_files = find_dicom_files(series_path)
        if not dicom_files:
            continue
        first_dicom = dicom_files[0][1]
        modality = getattr(first_dicom, 'Modality', '')
        series_desc = getattr(first_dicom, 'SeriesDescription', '')
    
        if modality == "CT":
            ct_series_path = series_path
        elif modality == "MR" and "CET1" in series_desc:
            mr_t2_series_path = series_path
        elif modality == "RTSTRUCT":
            for filepath, ds in dicom_files:
                rtstruct_files.append((filepath, ds))
                rtstruct_data_list.append(ds)
        elif modality == "RTPLAN":
            for filepath, ds in dicom_files:
                rtplan_files.append((filepath, ds))
        elif modality == "RTDOSE":
            if len(dicom_files) == 1:
                rtdose_files.append(dicom_files[0])
            else:
                rtdose_series_paths.append(series_path)
    
    # Validate required series
    if mr_t2_series_path is None:
        raise FileNotFoundError("No MR T2 series found")
    else:
        logging.info("MR T2 series path found: %s", mr_t2_series_path)
    
    if ct_series_path is None:
        raise FileNotFoundError("No CT series found")
    else:
        logging.info("CT series path found: %s", ct_series_path)
    
    rtstruct_data_list = [ds for (path, ds) in rtstruct_files if not os.path.basename(path).startswith('._')]
    if not rtstruct_data_list:
        raise FileNotFoundError("No RTSTRUCT files found")
    else:
        logging.info("%d RTSTRUCT file(s) found.", len(rtstruct_data_list))
    
    if not rtplan_files:
        raise FileNotFoundError("No RTPLAN files found")
    else:
        logging.info("%d RTPLAN file(s) found.", len(rtplan_files))
    
    if not (rtdose_files or rtdose_series_paths):
        raise FileNotFoundError("No RTDOSE data found")
    else:
        logging.info("RTDOSE data found.")
    
    # Load CT and MR images
    logging.info("Loading CT DICOM series...")
    ct_image_sitk = load_dicom_series(ct_series_path)
    if ct_image_sitk is None:
        raise FileNotFoundError(f"Could not load CT series from {ct_series_path}")
    
    logging.info("Loading MR DICOM series...")
    mr_image_sitk = load_dicom_series(mr_t2_series_path)
    if mr_image_sitk is None:
        raise FileNotFoundError(f"Could not load MR series from {mr_t2_series_path}")
    
    # Register MR to CT
    logging.info("Registering MR image to CT image...")
    registered_mr_sitk, final_transform = register_mr_to_ct(ct_image_sitk, mr_image_sitk)
    logging.info("Registration completed.")
    
    # Create mapping from CT slices (by Z position) to slice indices
    ct_files = find_dicom_files(ct_series_path)
    ct_files.sort(key=lambda x: x[1].ImagePositionPatient[2])
    ct_slice_z_positions = [(idx, ds.ImagePositionPatient[2]) for idx, (f, ds) in enumerate(ct_files)]
    
    # Process RTSTRUCT contours
    slice_contours: Dict[int, List[Dict[str, Any]]] = {}
    rois_without_contours: List[str] = []
    all_structure_names: Set[str] = set()
    
    for rtstruct in rtstruct_data_list:
        if not hasattr(rtstruct, 'StructureSetROISequence'):
            logging.warning("RTSTRUCT missing 'StructureSetROISequence'. Skipping dataset.")
            continue
    
        roi_dict = {}
        for roi in rtstruct.StructureSetROISequence:
            if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                roi_dict[roi.ROINumber] = roi.ROIName
            else:
                logging.warning("ROI missing 'ROINumber' or 'ROIName'.")
    
        roi_numbers_with_contours = set()
        if hasattr(rtstruct, 'ROIContourSequence'):
            for roi_contour in rtstruct.ROIContourSequence:
                if hasattr(roi_contour, 'ReferencedROINumber'):
                    roi_numbers_with_contours.add(roi_contour.ReferencedROINumber)
                else:
                    logging.warning("ROIContourSequence item missing 'ReferencedROINumber'.")
        else:
            logging.warning("RTSTRUCT has no ROIContourSequence.")
            continue
    
        for roi_number, roi_name in roi_dict.items():
            if roi_number in roi_numbers_with_contours:
                for contour_seq in rtstruct.ROIContourSequence:
                    if hasattr(contour_seq, 'ReferencedROINumber') and contour_seq.ReferencedROINumber == roi_number:
                        all_structure_names.add(roi_name)
                        if hasattr(contour_seq, 'ContourSequence'):
                            for contour in contour_seq.ContourSequence:
                                contour_data = getattr(contour, 'ContourData', [])
                                if not contour_data:
                                    logging.warning("Skipping empty ContourData for '%s'", roi_name)
                                    continue
                                try:
                                    contour_points = np.array(contour_data).reshape(-1, 3)
                                except ValueError:
                                    logging.warning("Invalid ContourData format for '%s'", roi_name)
                                    continue
                                mean_contour_z = np.mean(contour_points[:, 2])
                                slice_found = False
                                for idx, z_pos in ct_slice_z_positions:
                                    if np.isclose(z_pos, mean_contour_z, atol=1e-3):
                                        slice_contours.setdefault(idx, []).append({
                                            'points': contour_points,
                                            'structure_name': roi_name
                                        })
                                        logging.info("Mapped structure '%s' to slice %d (Z=%.2f mm)",
                                                     roi_name, idx, z_pos)
                                        slice_found = True
                                        break
                                if not slice_found:
                                    logging.warning("Could not map '%s' to any slice.", roi_name)
            else:
                logging.info("ROI '%s' (ROINumber %s) has no contour data.", roi_name, roi_number)
                rois_without_contours.append(roi_name)
    
    slices_with_structures = sorted(slice_contours.keys())
    logging.info("CT slices with structures: %s", slices_with_structures)
    
    # Load RTDOSE data
    logging.info("Loading RTDOSE...")
    dose_image_sitk, rtdose_dataset = load_rtdose_data(rtdose_files, rtdose_series_paths, ct_image_sitk)
    if dose_image_sitk is None:
        logging.warning("Failed to load RTDOSE data. Creating empty dose image.")
        dose_image_sitk = sitk.Image(ct_image_sitk.GetSize(), sitk.sitkFloat32)
        dose_image_sitk.CopyInformation(ct_image_sitk)
        dose_array = np.zeros(sitk.GetArrayFromImage(dose_image_sitk).shape, dtype=np.float32)
        dose_grid_scaling = 1.0
        dose_units = 'Unknown'
        dose_type = 'Unknown'
    else:
        logging.info("RTDOSE loaded.")
    
        same_origin = np.allclose(dose_image_sitk.GetOrigin(), ct_image_sitk.GetOrigin())
        same_direction = np.allclose(dose_image_sitk.GetDirection(), ct_image_sitk.GetDirection())
        logging.info("Dose Origin match: %s, Direction match: %s", same_origin, same_direction)
    
        def resample_dose_to_ct(ct_img: sitk.Image, dose_img: sitk.Image) -> sitk.Image:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ct_img)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform())
            return resampler.Execute(dose_img)
    
        if not (same_origin and same_direction):
            logging.info("Resampling RTDOSE to CT grid...")
            dose_image_sitk = resample_dose_to_ct(ct_image_sitk, dose_image_sitk)
            logging.info("Dose resampled.")
        else:
            logging.info("RTDOSE already in CT grid.")
    
        dose_array = sitk.GetArrayFromImage(dose_image_sitk).astype(np.float32)
        try:
            dose_grid_scaling = float(getattr(rtdose_dataset, 'DoseGridScaling', 1.0))
            logging.info("DoseGridScaling: %f", dose_grid_scaling)
        except Exception as e:
            logging.error("Error reading DoseGridScaling: %s", e)
            dose_grid_scaling = 1.0
        dose_units = getattr(rtdose_dataset, 'DoseUnits', 'Unknown')
        dose_type = getattr(rtdose_dataset, 'DoseType', 'Unknown')
        logging.info("RTDOSE DoseUnits: %s, DoseType: %s", dose_units, dose_type)
        dose_array *= dose_grid_scaling
    
    # Convert images to numpy arrays
    ct_array = sitk.GetArrayFromImage(ct_image_sitk).astype(np.float32)
    mr_array = sitk.GetArrayFromImage(registered_mr_sitk).astype(np.float32)
    
    # Downsample arrays
    logging.info("Downsampling volumetric data to reduce memory usage...")
    downsample_factor = 2
    ct_array_ds = downsample_volume(ct_array, factor=downsample_factor)
    mr_array_ds = downsample_volume(mr_array, factor=downsample_factor)
    dose_array_ds = downsample_volume(dose_array, factor=downsample_factor)
    logging.info("Downsampling factor: %d", downsample_factor)
    logging.info("Original CT shape: %s, Downsampled: %s", ct_array.shape, ct_array_ds.shape)
    logging.info("Original MR shape: %s, Downsampled: %s", mr_array.shape, mr_array_ds.shape)
    logging.info("Original Dose shape: %s, Downsampled: %s", dose_array.shape, dose_array_ds.shape)
    
    # Ensure same shape
    min_shape = np.minimum(ct_array_ds.shape, np.minimum(mr_array_ds.shape, dose_array_ds.shape))
    ct_array_ds = ct_array_ds[:min_shape[0], :min_shape[1], :min_shape[2]]
    mr_array_ds = mr_array_ds[:min_shape[0], :min_shape[1], :min_shape[2]]
    dose_array_ds = dose_array_ds[:min_shape[0], :min_shape[1], :min_shape[2]]
    logging.info("Final downsampled shape: %s", min_shape)
    
    # Extract beam geometries from RTPLAN
    if rtplan_files:
        rtplan_dataset = rtplan_files[0][1]
        beam_geometries = get_beam_geometries(rtplan_dataset)
        logging.info("Extracted beam geometries from RTPLAN: %d beams found.", len(beam_geometries))
    else:
        logging.warning("No RTPLAN files available for beam geometry extraction.")
        beam_geometries = []
    
    # Create 3D visualization
    fig = create_3d_visualization(ct_array_ds, mr_array_ds, dose_array_ds,
                                  slice_contours, beam_geometries,
                                  all_structure_names, rois_without_contours)
    fig.show()


if __name__ == '__main__':
    main()
