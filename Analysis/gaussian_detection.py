import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse


def detect_and_plot_gaussian_blobs(
    calibrated_img_array: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    num_sigma: int = 10,
    threshold: float = 0.2,
    figsize=(10, 8),
    show: bool = True
):
    """
    Detect Gaussian blobs in a calibrated image using Laplacian of Gaussian (LoG)
    and optionally plot the results.

    Parameters
    ----------
    calibrated_img_array : np.ndarray
        Input calibrated image (2D array).
    min_sigma : float, optional
        Minimum standard deviation for Gaussian kernel.
    max_sigma : float, optional
        Maximum standard deviation for Gaussian kernel.
    num_sigma : int, optional
        Number of intermediate sigma values.
    threshold : float, optional
        Absolute lower bound for scale-space maxima.
    figsize : tuple, optional
        Figure size for visualization.
    show : bool, optional
        Whether to display the image with detected blobs.

    Returns
    -------
    np.ndarray
        Array of detected blobs with shape (N, 3),
        where each blob is (row, col, sigma).
    """

    if calibrated_img_array is None:
        raise ValueError("calibrated_img_array is None.")

    # Detect blobs using Laplacian of Gaussian
    gaussian_blobs = blob_log(
        calibrated_img_array,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )

    print(f"Detected {len(gaussian_blobs)} Gaussian peaks.")

    if show and len(gaussian_blobs) > 0:
        rows = gaussian_blobs[:, 0]
        cols = gaussian_blobs[:, 1]
        sigmas = gaussian_blobs[:, 2]

        plt.figure(figsize=figsize)
        plt.imshow(calibrated_img_array, cmap='gray')

        # Draw circles for each detected blob
        ax = plt.gca()
        for y, x, r in gaussian_blobs:
            circle = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
            ax.add_patch(circle)

        # Optional scatter of centers
        plt.scatter(
            cols,
            rows,
            s=sigmas * 5,
            c='red',
            alpha=0.6,
            edgecolors='none',
            label='Detected Gaussian Peaks'
        )

        plt.title('Calibrated Image with Detected Gaussian Peaks (blob_log)')
        plt.axis('off')
        plt.legend()
        plt.show()

    return gaussian_blobs


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    Returns a 2D Gaussian function for curve fitting.

    Parameters:
    -----------
    coords : tuple
        A tuple (x, y) where x and y are 1D arrays of coordinates.
    amplitude : float
        The amplitude of the Gaussian peak.
    x0 : float
        The x-coordinate of the center of the Gaussian.
    y0 : float
        The y-coordinate of the center of the Gaussian.
    sigma_x : float
        The standard deviation of the Gaussian in the x-direction.
    sigma_y : float
        The standard deviation of the Gaussian in the y-direction.
    offset : float
        The background offset.

    Returns:
    --------
    np.ndarray
        A 1D array representing the 2D Gaussian values, flattened.
    """
    x, y = coords

    # Calculate the 2D Gaussian formula
    exponent = -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    g = offset + amplitude * np.exp(exponent)

    return g.ravel() # Ensure the output is a flattened 1D array

def fit_2d_gaussians(
    calibrated_img_array: np.ndarray,
    gaussian_blobs: np.ndarray,
    gaussian_2d,
    roi_sigma_factor: float = 5.0,
    sigma_lower_factor: float = 0.1,
    sigma_upper_factor: float = 10.0,
    figsize=(10, 8),
    verbose: bool = True
):
    """
    Fit 2D Gaussians to detected blobs and plot a single final overlay
    of all fitted Gaussians.

    Returns
    -------
    list of dict
        Fitted Gaussian parameters.
    """

    fitted_gaussians = []
    img_h, img_w = calibrated_img_array.shape

    for i, (y_c, x_c, sigma) in enumerate(gaussian_blobs):
        y_c = int(round(y_c))
        x_c = int(round(x_c))

        roi_half = int(sigma * roi_sigma_factor)

        y_min = max(0, y_c - roi_half)
        y_max = min(img_h, y_c + roi_half)
        x_min = max(0, x_c - roi_half)
        x_max = min(img_w, x_c + roi_half)

        if (y_max - y_min <= 1) or (x_max - x_min <= 1):
            continue

        roi = calibrated_img_array[y_min:y_max, x_min:x_max]

        x_roi = np.arange(x_min, x_max)
        y_roi = np.arange(y_min, y_max)
        X, Y = np.meshgrid(x_roi, y_roi)

        p0 = [
            np.max(roi) - np.min(roi),
            x_c,
            y_c,
            sigma,
            sigma,
            np.min(roi)
        ]

        bounds = (
            [0, x_min, y_min, sigma * sigma_lower_factor, sigma * sigma_lower_factor, 0],
            [np.inf, x_max, y_max, sigma * sigma_upper_factor, sigma * sigma_upper_factor, np.max(calibrated_img_array)]
        )

        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (X.ravel(), Y.ravel()),
                roi.ravel(),
                p0=p0,
                bounds=bounds
            )

            fitted_gaussians.append({
                'amplitude': popt[0],
                'x0': popt[1],
                'y0': popt[2],
                'sigma_x': popt[3],
                'sigma_y': popt[4],
                'offset': popt[5]
            })

        except (RuntimeError, ValueError):
            continue

    # ---------------- FINAL OVERLAY PLOT ----------------
    if verbose:
      fig, ax = plt.subplots(figsize=figsize)
      ax.imshow(calibrated_img_array, cmap='gray')

      for g in fitted_gaussians:
          # Center
          ax.scatter(g['x0'], g['y0'], c='red', s=5)

          # 1σ ellipse
          ellipse = Ellipse(
              (g['x0'], g['y0']),
              width=2 * g['sigma_x'],
              height=2 * g['sigma_y'],
              edgecolor='green',
              facecolor='none',
              linewidth=1
          )
          ax.add_patch(ellipse)

      ax.set_title(f'Final Overlay: {len(fitted_gaussians)} Fitted 2D Gaussians')
      ax.axis('off')
      plt.show()

      if verbose:
          print(f"Successfully fitted and plotted {len(fitted_gaussians)} Gaussians.")

    return fitted_gaussians
