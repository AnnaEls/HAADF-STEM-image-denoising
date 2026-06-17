#####################################################################
# Strain mapping is based on the method developed in:
# Xiaonan Luo, Aakash Varambhia, Weixin Song, Dogan Ozkaya, Sergio Lozano-Perez, Peter D. Nellist,
#High-precision atomic-scale strain mapping of nanoparticles from STEM images,
#Ultramicroscopy, Volume 239, 2022, 113561, ISSN 0304-3991,
#https://doi.org/10.1016/j.ultramic.2022.113561.
#(https://www.sciencedirect.com/science/article/pii/S0304399122000936)
#################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import polar
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

import tifffile


# ============================================================
# Local atomic strain estimation
# ============================================================

def reference_neighbor_vectors(v1, v2):
    """
    Six nearest-neighbor vectors of a triangular Bravais lattice.

    v1 and v2 must be primitive lattice vectors with an angle
    close to 60 degrees.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    return np.array([
        v1,
        v2,
        v2 - v1,
        -v1,
        -v2,
        v1 - v2
    ])


def weighted_deformation_gradient(reference_vectors,
                                  measured_vectors,
                                  weights=None):
    """
    Fit the local deformation gradient F from

        measured_vector ≈ F @ reference_vector

    Parameters
    ----------
    reference_vectors : (M, 2)
        Ideal lattice vectors.

    measured_vectors : (M, 2)
        Corresponding measured displacement vectors.

    weights : (M,), optional
        Fitting weights.

    Returns
    -------
    F : (2, 2)
        Local deformation gradient.
    residual_rmse : float
        RMS vector fitting error.
    """
    X = np.asarray(reference_vectors, dtype=float)
    Y = np.asarray(measured_vectors, dtype=float)

    if len(X) < 2:
        raise ValueError("At least two independent vectors are required.")

    if weights is None:
        weights = np.ones(len(X), dtype=float)

    weights = np.asarray(weights, dtype=float)
    W = np.diag(weights)

    # Row-vector representation:
    #
    # Y ≈ X @ F.T
    #
    # F.T = (X.T W X)^(-1) X.T W Y
    normal_matrix = X.T @ W @ X

    if np.linalg.cond(normal_matrix) > 1e10:
        raise np.linalg.LinAlgError(
            "Neighbor configuration is ill-conditioned."
        )

    F_transpose = np.linalg.solve(
        normal_matrix,
        X.T @ W @ Y
    )

    F = F_transpose.T

    predicted = X @ F.T
    residual_rmse = np.sqrt(
        np.mean(
            np.sum((Y - predicted) ** 2, axis=1)
        )
    )

    return F, residual_rmse


def calculate_strain_from_F(F):
    """
    Calculate rotation-independent strain quantities.

    Polar decomposition:

        F = R @ U

    where
        R = local rigid-body rotation
        U = right stretch tensor

    Biot strain:

        strain = U - I

    Principal strains:

        principal_strain = eigenvalue(U) - 1
    """
    F = np.asarray(F, dtype=float)

    # scipy.linalg.polar with side='right':
    # F = R @ U
    R, U = polar(F, side="right")

    strain = U - np.eye(2)

    eigenvalues, eigenvectors = np.linalg.eigh(U)

    # Sort from maximum to minimum principal stretch
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    principal_strains = eigenvalues - 1.0

    rotation = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    # Green-Lagrange strain is also returned for comparison
    green_lagrange = 0.5 * (
        F.T @ F - np.eye(2)
    )

    return {
        "rotation_matrix": R,
        "stretch_tensor": U,
        "biot_strain": strain,
        "green_lagrange_strain": green_lagrange,
        "principal_strain_max": principal_strains[0],
        "principal_strain_min": principal_strains[1],
        "principal_direction_max": eigenvectors[:, 0],
        "principal_direction_min": eigenvectors[:, 1],
        "rotation_rad": rotation,
        "rotation_deg": np.degrees(rotation),
        "area_ratio": np.linalg.det(F)
    }


def estimate_atomic_strain(
    points_xy,
    v1,
    v2,
    neighbor_radius_factor=1.40,
    matching_tolerance_factor=0.35,
    minimum_matches=3,
    gaussian_weight_sigma_factor=0.25,
    convert_image_to_cartesian=True
):
    """
    Estimate local strain at every atomic-column position.

    Parameters
    ----------
    points_xy : (N, 2)
        Atomic coordinates in pixels as [x, y].

    v1, v2 : array-like, shape (2,)
        Reference primitive lattice vectors in pixels.

    neighbor_radius_factor : float
        Search radius relative to the reference lattice spacing.

    matching_tolerance_factor : float
        Maximum allowed mismatch between a measured vector
        and its assigned reference vector, relative to a0.

    minimum_matches : int
        Minimum number of matched neighbor vectors.

    gaussian_weight_sigma_factor : float
        Width used for distance weighting of neighbor-vector
        residuals.

    convert_image_to_cartesian : bool
        If True, convert image coordinates (y downward) to
        Cartesian coordinates (y upward).

    Returns
    -------
    results : pandas.DataFrame
        Local strain and deformation parameters for every
        successfully analyzed atomic column.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")

    # Convert image coordinates:
    # x -> right
    # y -> upward
    #
    # This only changes signs of shear and rotation-related terms.
    if convert_image_to_cartesian:
        transform = np.array([
            [1.0,  0.0],
            [0.0, -1.0]
        ])

        points_work = points_xy @ transform.T
        v1_work = transform @ v1
        v2_work = transform @ v2
    else:
        points_work = points_xy.copy()
        v1_work = v1.copy()
        v2_work = v2.copy()

    reference_vectors = reference_neighbor_vectors(
        v1_work,
        v2_work
    )

    reference_lengths = np.linalg.norm(
        reference_vectors,
        axis=1
    )

    a0 = np.mean(reference_lengths)

    neighbor_radius = neighbor_radius_factor * a0
    matching_tolerance = matching_tolerance_factor * a0
    weight_sigma = gaussian_weight_sigma_factor * a0

    tree = cKDTree(points_work)

    rows = []

    for center_index, center in enumerate(points_work):

        neighbor_indices = tree.query_ball_point(
            center,
            r=neighbor_radius
        )

        neighbor_indices = [
            index for index in neighbor_indices
            if index != center_index
        ]

        if len(neighbor_indices) < minimum_matches:
            continue

        measured_vectors_all = (
            points_work[neighbor_indices] - center
        )

        measured_lengths = np.linalg.norm(
            measured_vectors_all,
            axis=1
        )

        # Exclude unrealistically short vectors
        valid_length = measured_lengths > 0.55 * a0

        measured_vectors_all = measured_vectors_all[valid_length]
        neighbor_indices = np.asarray(neighbor_indices)[valid_length]

        if len(measured_vectors_all) < minimum_matches:
            continue

        # ----------------------------------------------------
        # Match measured vectors to ideal reference vectors
        # ----------------------------------------------------

        # Cost matrix:
        # rows    = measured vectors
        # columns = six reference vectors
        cost = np.linalg.norm(
            measured_vectors_all[:, None, :]
            - reference_vectors[None, :, :],
            axis=2
        )

        measured_assignment, reference_assignment = (
            linear_sum_assignment(cost)
        )

        assignment_errors = cost[
            measured_assignment,
            reference_assignment
        ]

        accepted = assignment_errors <= matching_tolerance

        measured_assignment = measured_assignment[accepted]
        reference_assignment = reference_assignment[accepted]
        assignment_errors = assignment_errors[accepted]

        if len(measured_assignment) < minimum_matches:
            continue

        measured_vectors = measured_vectors_all[
            measured_assignment
        ]

        matched_reference_vectors = reference_vectors[
            reference_assignment
        ]

        # Require two independent reference directions
        if np.linalg.matrix_rank(
            matched_reference_vectors
        ) < 2:
            continue

        # More accurately matched vectors receive higher weights
        weights = np.exp(
            -0.5 * (
                assignment_errors / weight_sigma
            ) ** 2
        )

        try:
            F, fit_rmse = weighted_deformation_gradient(
                matched_reference_vectors,
                measured_vectors,
                weights=weights
            )

            strain_result = calculate_strain_from_F(F)

        except (ValueError, np.linalg.LinAlgError):
            continue

        strain = strain_result["biot_strain"]
        green = strain_result["green_lagrange_strain"]

        # Direct local lattice parameter based on matched vectors
        measured_neighbor_lengths = np.linalg.norm(
            measured_vectors,
            axis=1
        )

        reference_neighbor_lengths = np.linalg.norm(
            matched_reference_vectors,
            axis=1
        )

        local_scale_factors = (
            measured_neighbor_lengths
            / reference_neighbor_lengths
        )

        local_lattice_parameter_px = (
            a0 * np.mean(local_scale_factors)
        )

        local_lattice_strain = (
            local_lattice_parameter_px / a0 - 1.0
        )

        # Area-equivalent isotropic lattice parameter:
        #
        # det(F) is the local area ratio, so sqrt(det(F))
        # is the corresponding 2D linear scale factor.
        area_ratio = strain_result["area_ratio"]

        if area_ratio > 0:
            area_equivalent_lattice_parameter_px = (
                a0 * np.sqrt(area_ratio)
            )
        else:
            area_equivalent_lattice_parameter_px = np.nan

        max_direction = strain_result[
            "principal_direction_max"
        ]

        min_direction = strain_result[
            "principal_direction_min"
        ]

        rows.append({
            "atom_index": center_index,

            # Original image coordinates
            "x_px": points_xy[center_index, 0],
            "y_px": points_xy[center_index, 1],

            "number_of_matches": len(measured_vectors),
            "matching_rmse_px": np.sqrt(
                np.mean(assignment_errors ** 2)
            ),
            "deformation_fit_rmse_px": fit_rmse,

            # Deformation gradient
            "F_xx": F[0, 0],
            "F_xy": F[0, 1],
            "F_yx": F[1, 0],
            "F_yy": F[1, 1],

            # Biot strain U - I
            "strain_xx": strain[0, 0],
            "strain_yy": strain[1, 1],
            "strain_xy": strain[0, 1],

            # Green-Lagrange strain
            "green_xx": green[0, 0],
            "green_yy": green[1, 1],
            "green_xy": green[0, 1],

            # Principal strains
            "principal_strain_max": strain_result[
                "principal_strain_max"
            ],
            "principal_strain_min": strain_result[
                "principal_strain_min"
            ],

            "principal_max_direction_x": max_direction[0],
            "principal_max_direction_y": max_direction[1],

            "principal_min_direction_x": min_direction[0],
            "principal_min_direction_y": min_direction[1],

            "rotation_deg": strain_result["rotation_deg"],
            "area_ratio": area_ratio,

            # Reference-independent or less reference-sensitive
            # lattice measures
            "reference_lattice_parameter_px": a0,
            "local_lattice_parameter_px":
                local_lattice_parameter_px,
            "local_lattice_strain":
                local_lattice_strain,
            "area_equivalent_lattice_parameter_px":
                area_equivalent_lattice_parameter_px
        })

    return pd.DataFrame(rows)


# ============================================================
# Plot atomic strain values
# ============================================================

def plot_atomic_quantity(
    image,
    results,
    quantity,
    title=None,
    cmap="RdBu_r",
    symmetric=True,
    marker_size=45,
    percent=False,
    xlim=None,
    ylim=None,
    save_path=None
):
    """
    Plot a strain quantity directly at atomic-column positions.
    """
    values = results[quantity].to_numpy().copy()

    if percent:
        values *= 100.0

    if symmetric:
        limit = np.nanpercentile(
            np.abs(values),
            98
        )
        vmin = -limit
        vmax = limit
    else:
        vmin = np.nanpercentile(values, 2)
        vmax = np.nanpercentile(values, 98)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(
        image,
        cmap="gray",
        origin="upper"
    )

    scatter = ax.scatter(
        results["x_px"],
        results["y_px"],
        c=values,
        s=marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.15
    )

    colorbar = plt.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    if percent:
        colorbar.set_label("%")
    else:
        colorbar.set_label(quantity)

    if title is None:
        title = quantity

    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        # Image coordinates increase downward
        ax.set_ylim(ylim[1], ylim[0])

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=250,
            bbox_inches="tight"
        )

    plt.show()


def interpolate_atomic_quantity(
    image_shape,
    results,
    quantity,
    grid_step=1,
    interpolation_method="linear"
):
    """
    Interpolate an atomic strain quantity onto an image grid.

    Interpolation does not add physical resolution. It is only
    used for visualization.
    """
    height, width = image_shape

    grid_x, grid_y = np.meshgrid(
        np.arange(0, width, grid_step),
        np.arange(0, height, grid_step)
    )

    values = griddata(
        points=results[["x_px", "y_px"]].to_numpy(),
        values=results[quantity].to_numpy(),
        xi=(grid_x, grid_y),
        method=interpolation_method
    )

    return grid_x, grid_y, values


def plot_interpolated_quantity(
    image,
    results,
    quantity,
    title=None,
    cmap="RdBu_r",
    percent=False,
    symmetric=True,
    alpha=0.75,
    xlim=None,
    ylim=None,
    save_path=None
):
    """
    Plot an interpolated strain field over the STEM image.
    """
    _, _, field = interpolate_atomic_quantity(
        image.shape,
        results,
        quantity
    )

    if percent:
        field = field * 100.0

    finite = np.isfinite(field)

    if symmetric:
        limit = np.nanpercentile(
            np.abs(field[finite]),
            98
        )
        vmin = -limit
        vmax = limit
    else:
        vmin = np.nanpercentile(field[finite], 2)
        vmax = np.nanpercentile(field[finite], 98)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(
        image,
        cmap="gray",
        origin="upper"
    )

    overlay = ax.imshow(
        field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        origin="upper"
    )

    colorbar = plt.colorbar(
        overlay,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    colorbar.set_label("%" if percent else quantity)

    if title is None:
        title = quantity

    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim[1], ylim[0])

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=250,
            bbox_inches="tight"
        )

    plt.show()
