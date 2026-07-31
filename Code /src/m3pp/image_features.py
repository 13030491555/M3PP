from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def illumination_correct(gray: np.ndarray, sigma: float = 51.0) -> np.ndarray:
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = gray.astype(np.float32) - background.astype(np.float32)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def dark_phase_mask(gray: np.ndarray) -> np.ndarray:
    corrected = illumination_correct(gray)
    _, mask = cv2.threshold(
        corrected,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def white_area_fraction(gray: np.ndarray) -> float:
    corrected = illumination_correct(gray)
    _, mask = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float((mask > 0).mean() * 100.0)


def filter_objects(
    mask: np.ndarray,
    min_area: float = 50,
    min_circularity: float = 0.3,
    min_solidity: float = 0.5,
) -> np.ndarray:
    labeled = label(mask)
    retained = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labeled):
        perimeter = max(region.perimeter, 1e-12)
        circularity = 4.0 * np.pi * region.area / perimeter**2
        solidity = float(region.solidity or 0.0)
        if (
            region.area > min_area
            and circularity > min_circularity
            and solidity > min_solidity
        ):
            retained[labeled == region.label] = True
    return retained


def split_touching(mask: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(mask)
    coordinates = peak_local_max(
        distance,
        labels=mask,
        min_distance=3,
        exclude_border=False,
    )
    markers = np.zeros_like(mask, dtype=int)
    for marker_id, (row, column) in enumerate(coordinates, start=1):
        markers[row, column] = marker_id
    if markers.max() == 0:
        markers = label(mask)
    return watershed(-distance, markers, mask=mask)


def graphite_descriptors(gray: np.ndarray, pixel_size_um: float = 1.0) -> dict:
    filtered = filter_objects(dark_phase_mask(gray))
    segmented = split_touching(filtered)

    radii = []
    centroids = []
    retained = np.zeros_like(filtered, dtype=bool)
    for region in regionprops(segmented):
        perimeter = max(region.perimeter, 1e-12)
        circularity = 4.0 * np.pi * region.area / perimeter**2
        solidity = float(region.solidity or 0.0)
        if region.area <= 50 or circularity <= 0.3 or solidity <= 0.5:
            continue

        coordinates = np.column_stack(np.nonzero(segmented == region.label))
        points = coordinates[:, ::-1].astype(np.float32)
        (_, _), radius_px = cv2.minEnclosingCircle(points)
        radii.append(float(radius_px * pixel_size_um))
        centroids.append(
            (
                float(region.centroid[0] * pixel_size_um),
                float(region.centroid[1] * pixel_size_um),
            )
        )
        retained[segmented == region.label] = True

    if not radii:
        raise ValueError("No graphite objects passed the filtering criteria.")

    radii_array = np.asarray(radii, dtype=float)
    if len(centroids) > 1:
        distances = cdist(centroids, centroids)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
    else:
        nearest = np.asarray([0.0])

    return {
        "average_graphite_radius": float(radii_array.mean()),
        "std_graphite_radius": float(radii_array.std(ddof=0)),
        "average_nearest_neighbor_distance": float(nearest.mean()),
        "std_nearest_neighbor_distance": float(nearest.std(ddof=0)),
        "graphite_area_fraction": float(retained.mean() * 100.0),
    }


def paired_descriptors(
    unetched_path: Path,
    etched_path: Path,
    pixel_size_um: float = 1.0,
) -> dict:
    unetched = cv2.imread(str(unetched_path), cv2.IMREAD_GRAYSCALE)
    etched = cv2.imread(str(etched_path), cv2.IMREAD_GRAYSCALE)
    if unetched is None or etched is None:
        raise FileNotFoundError("Unable to read one or both image files.")

    descriptors = graphite_descriptors(unetched, pixel_size_um)
    pearlite_descriptor = white_area_fraction(etched) - descriptors["graphite_area_fraction"]
    descriptors["image_derived_pearlite_descriptor"] = float(pearlite_descriptor)
    return descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unetched", required=True)
    parser.add_argument("--etched", required=True)
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    result = paired_descriptors(
        Path(arguments.unetched),
        Path(arguments.etched),
        arguments.pixel_size_um,
    )
    Path(arguments.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
