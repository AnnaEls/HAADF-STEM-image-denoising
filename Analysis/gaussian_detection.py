import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from skimage.feature import peak_local_max


def fit_local_gaussian(image, y_init, x_init, radius=5):
    """
    Fit an axis-aligned 2D Gaussian around one detected peak.

    Returns
    -------
    x_center, y_center, amplitude, sigma_x, sigma_y, background, rmse
    """
    height, width = image.shape

    x_int = int(round(x_init))
    y_int = int(round(y_init))

    x_min = max(0, x_int - radius)
    x_max = min(width, x_int + radius + 1)
    y_min = max(0, y_int - radius)
    y_max = min(height, y_int + radius + 1)

    patch = image[y_min:y_max, x_min:x_max]

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    background_0 = np.percentile(patch, 10)
    amplitude_0 = max(patch.max() - background_0, 1e-12)

    weights = np.clip(patch - background_0, 0, None)

    if weights.sum() > 0:
        x_center_0 = np.sum(weights * xx) / np.sum(weights)
        y_center_0 = np.sum(weights * yy) / np.sum(weights)
    else:
        x_center_0 = x_init
        y_center_0 = y_init

    initial_parameters = np.array([
        amplitude_0,
        x_center_0,
        y_center_0,
        2.0,                  # sigma_x
        2.0,                  # sigma_y
        background_0
    ])

    def gaussian_model(parameters):
        amplitude, x_center, y_center, sigma_x, sigma_y, background = parameters

        gaussian = amplitude * np.exp(
            -0.5 * (
                ((xx - x_center) / sigma_x) ** 2
                + ((yy - y_center) / sigma_y) ** 2
            )
        )

        return background + gaussian

    def residual(parameters):
        return (gaussian_model(parameters) - patch).ravel()

    lower_bounds = [
        0,
        max(x_min - 0.5, x_init - 3),
        max(y_min - 0.5, y_init - 3),
        0.5,
        0.5,
        -np.inf
    ]

    upper_bounds = [
        np.inf,
        min(x_max - 0.5, x_init + 3),
        min(y_max - 0.5, y_init + 3),
        6.0,
        6.0,
        np.inf
    ]

    result = least_squares(
        residual,
        initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=200
    )

    amplitude, x_center, y_center, sigma_x, sigma_y, background = result.x

    rmse = np.sqrt(np.mean(residual(result.x) ** 2))

    return {
        "x_px": x_center,
        "y_px": y_center,
        "amplitude": amplitude,
        "sigma_x_px": sigma_x,
        "sigma_y_px": sigma_y,
        "background": background,
        "fit_rmse": rmse
    }


def locate_gaussians(
    image,
    smoothing_sigma=1.0,
    min_distance=4,
    relative_threshold=0.08,
    fit_radius=5
):
    """
    Detect local maxima and refine each center using 2D Gaussian fitting.
    """
    image = np.squeeze(image).astype(np.float64)

    smoothed = gaussian_filter(image, sigma=smoothing_sigma)

    threshold = (
        smoothed.min()
        + relative_threshold * (smoothed.max() - smoothed.min())
    )

    initial_peaks_yx = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False
    )

    fitted_peaks = []

    for y_peak, x_peak in initial_peaks_yx:
        result = fit_local_gaussian(
            image,
            y_init=float(y_peak),
            x_init=float(x_peak),
            radius=fit_radius
        )

        fitted_peaks.append(result)

    coordinates = pd.DataFrame(fitted_peaks)
    coordinates.insert(0, "id", np.arange(1, len(coordinates) + 1))

    return coordinates

def plot_patch_with_overlay(
    image,
    coordinates,
    center_x=None,
    center_y=None,
    crop_size=180,
    marker_size=28,
    save_path=None
):
    """
    Display an image/image patch with fitted Gaussian centers.
    """
    image = np.squeeze(image)

    height, width = image.shape

    if center_x is None:
        center_x = width // 2

    if center_y is None:
        center_y = height // 2

    half_size = crop_size // 2

    x_min = max(0, int(center_x - half_size))
    x_max = min(width, int(center_x + half_size))

    y_min = max(0, int(center_y - half_size))
    y_max = min(height, int(center_y + half_size))

    patch = image[y_min:y_max, x_min:x_max]

    inside_patch = (
        (coordinates["x_px"] >= x_min)
        & (coordinates["x_px"] < x_max)
        & (coordinates["y_px"] >= y_min)
        & (coordinates["y_px"] < y_max)
    )

    patch_coordinates = coordinates.loc[inside_patch].copy()

    patch_coordinates["x_local"] = patch_coordinates["x_px"] - x_min
    patch_coordinates["y_local"] = patch_coordinates["y_px"] - y_min

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        patch,
        cmap="gray",
        interpolation="nearest"
    )

    ax.scatter(
        patch_coordinates["x_local"],
        patch_coordinates["y_local"],
        s=marker_size,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0
    )

    ax.set_title(
        f"Gaussian centers\n"
        f"x = {x_min}:{x_max}, y = {y_min}:{y_max}, "
        f"N = {len(patch_coordinates)}"
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=250,
            bbox_inches="tight"
        )

    plt.show()

    return patch_coordinates
