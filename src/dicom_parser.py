"""
DICOM Parser Module
-------------------
Reads DICOM (.dcm) medical images, extracts pixel data & metadata,
normalizes them for processing, and can save reconstructed images
back into valid DICOM format.
"""

import os
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime


def load_dicom(file_path):
    """
    Loads a DICOM image and its metadata.
    Args:
        file_path (str): Path to the .dcm file.
    Returns:
        tuple: (normalized_image (np.ndarray), metadata (dict))
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    dcm = pydicom.dcmread(file_path)
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Normalize to [0, 1]
    img = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-8)

    metadata = {
        "PatientID": getattr(dcm, "PatientID", "Unknown"),
        "StudyInstanceUID": getattr(dcm, "StudyInstanceUID", None),
        "SeriesInstanceUID": getattr(dcm, "SeriesInstanceUID", None),
        "SOPInstanceUID": getattr(dcm, "SOPInstanceUID", None),
        "Modality": getattr(dcm, "Modality", "Unknown"),
        "Rows": dcm.Rows,
        "Columns": dcm.Columns,
        "BitsAllocated": dcm.get("BitsAllocated", 16)
    }

    return img, metadata


def denormalize_image(img):
    """
    Converts normalized [0,1] image back to 16-bit scale.
    """
    img = np.clip(img, 0, 1)
    return (img * 65535).astype(np.uint16)


def save_dicom(img, metadata, save_path):
    """
    Saves an image and metadata back as a DICOM file.

    Args:
        img (np.ndarray): Image array (normalized or 16-bit)
        metadata (dict): Metadata dictionary from load_dicom
        save_path (str): Output .dcm file path
    """
    if img.dtype != np.uint16:
        img = denormalize_image(img)

    # Create minimal DICOM dataset
    dcm = FileDataset(save_path, {}, file_meta=Dataset(), preamble=b"\0" * 128)

    # Assign basic tags
    dcm.PatientID = metadata.get("PatientID", "Unknown")
    dcm.StudyInstanceUID = metadata.get("StudyInstanceUID", "1.2.3")
    dcm.SeriesInstanceUID = metadata.get("SeriesInstanceUID", "1.2.3.1")
    dcm.SOPInstanceUID = metadata.get("SOPInstanceUID", "1.2.3.1.1")
    dcm.Modality = metadata.get("Modality", "OT")

    dcm.Rows = img.shape[0]
    dcm.Columns = img.shape[1]
    dcm.SamplesPerPixel = 1
    dcm.PhotometricInterpretation = "MONOCHROME2"
    dcm.BitsAllocated = 16
    dcm.BitsStored = 16
    dcm.HighBit = 15
    dcm.PixelRepresentation = 0
    dcm.PixelData = img.tobytes()

    # Add timestamp and save
    dcm.ContentDate = datetime.now().strftime("%Y%m%d")
    dcm.ContentTime = datetime.now().strftime("%H%M%S.%f")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dcm.save_as(save_path)
    print(f"✅ Saved reconstructed DICOM at: {save_path}")


if __name__ == "__main__":
    # Test run on a single file (for sanity check)
    sample_path = "../data/siim_pneumothorax/stage_2_images/sample.dcm"
    if os.path.exists(sample_path):
        img, meta = load_dicom(sample_path)
        print("Loaded DICOM successfully")
        print(f"Metadata: {meta}")
        save_dicom(img, meta, "../outputs/test_reconstruct.dcm")
    else:
        print("⚠️ Sample DICOM not found. Place one in 'stage_2_images/'.")
