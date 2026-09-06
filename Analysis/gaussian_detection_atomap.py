import atomap.api as am
import hyperspy.api as hs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def find_atoms(image, separation, plot = False):
  atom_positions = am.get_atom_positions(image, separation=separation)
  sublattice = am.Sublattice(atom_positions, image=image.data)
  sublattice.find_nearest_neighbors()
  sublattice.refine_atom_positions_using_center_of_mass()
  sublattice.refine_atom_positions_using_2d_gaussian()
  if plot:
    sublattice.plot()
  return sublattice

def plot_atoms(sublattice, image, sublattice_ref=None, use_ref = False, color = 'red', s=30, full_plot = False):
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.imshow(image.data, cmap='gray')
  ax.axis('off')

  if use_ref:
        for idx in range(len(sublattice_ref.atom_list)):
          x_center_ref = sublattice_ref.x_position[idx]
          y_center_ref = sublattice_ref.y_position[idx]
          sigma_x_ref =  sublattice_ref.sigma_x[idx]
          sigma_y_ref = sublattice_ref.sigma_y[idx]
          rotation_angle_ref = math.degrees(sublattice_ref.rotation[idx])
          ellipse = Ellipse((x_center_ref, y_center_ref),
                            width=3 * sigma_x_ref,    # 2 * 2-sigma extent
                            height=3 * sigma_y_ref,   # 2 * 2-sigma extent
                            angle=rotation_angle_ref, # Angle in degrees
                            edgecolor='green',
                            facecolor='green',
                            linewidth=0.8,
                            alpha=0.5) # Semi-transparent
          if full_plot:
            ax.add_patch(ellipse)

  for idx in range(len(sublattice.atom_list)):
      x_center = sublattice.x_position[idx]
      y_center = sublattice.y_position[idx]
      sigma_x =  sublattice.sigma_x[idx]
      sigma_y = sublattice.sigma_y[idx]
      rotation_angle = math.degrees(sublattice.rotation[idx])
      ellipse = Ellipse((x_center, y_center),
                        width=3 * sigma_x,    # 2 * 2-sigma extent
                        height=3 * sigma_y,   # 2 * 2-sigma extent
                        angle=rotation_angle, # Angle in degrees
                        edgecolor=color,
                        facecolor=color,
                        linewidth=0.8,
                        alpha=0.3) # Semi-transparent

      if full_plot:
        ax.add_patch(ellipse)
      plt.scatter(x_center, y_center, color=color,marker='o', s=s)

  if use_ref:
      plt.scatter(sublattice_ref.x_position, sublattice_ref.y_position, color='green', marker='x', s=s)

  plt.tight_layout()
  plt.show()

def sublattice_to_xy(sublattice):
    """
    Convert a sublattice list of Atom_Position objects into an (N, 2) array
    containing only the Gaussian center positions.

    Parameters
    ----------
    sublattice : list
        List of Atom_Position objects.

    Returns
    -------
    xy : ndarray, shape (N, 2)
        Array of Gaussian center coordinates:
        [[x1, y1],
         [x2, y2],
         ...]
    """

    xy = np.array([[sublattice.x_position[idx], sublattice.y_position[idx]] for idx in range(len(sublattice.atom_list))], dtype=float)

    return xy

def estimate_lattice_vectors_from_points(xy, neighbor_radius=25):
    """
    Estimate two dominant lattice basis vectors from 2D point positions.

    This assumes a reasonably regular 2D lattice.
    """

    xy = np.asarray(xy, dtype=float)

    tree = cKDTree(xy)
    vectors = []

    for p in xy:
        idxs = tree.query_ball_point(p, r=neighbor_radius)

        for idx in idxs:
            q = xy[idx]
            v = q - p
            d = np.linalg.norm(v)

            if d > 1e-6:
                vectors.append(v)

    vectors = np.array(vectors)

    # Keep only short neighbor vectors
    lengths = np.linalg.norm(vectors, axis=1)
    median_len = np.median(lengths)
    short_vectors = vectors[lengths < 1.5 * median_len]

    # Separate mainly-horizontal and mainly-vertical directions
    abs_v = np.abs(short_vectors)

    horizontal = short_vectors[abs_v[:, 0] > abs_v[:, 1]]
    vertical = short_vectors[abs_v[:, 1] >= abs_v[:, 0]]

    # Orient vectors consistently
    horizontal = np.array([v if v[0] > 0 else -v for v in horizontal])
    vertical = np.array([v if v[1] > 0 else -v for v in vertical])

    a = np.median(horizontal, axis=0)
    b = np.median(vertical, axis=0)

    return a, b

def estimate_hexagonal_lattice_vectors_from_points(
    xy,
    neighbor_radius=25,
    angle_tolerance_deg=15
):
    """
    Estimate primitive lattice vectors for a 2D hexagonal/triangular lattice.

    Returns two nearest-neighbor lattice vectors a and b
    with approximately equal length and ~60 degree angle.
    """

    xy = np.asarray(xy, dtype=float)

    tree = cKDTree(xy)
    vectors = []

    # --------------------------------------------------
    # Collect neighbor displacement vectors
    # --------------------------------------------------
    for p in xy:

        idxs = tree.query_ball_point(p, r=neighbor_radius)

        for idx in idxs:

            q = xy[idx]
            v = q - p

            d = np.linalg.norm(v)

            if d > 1e-6:
                vectors.append(v)

    vectors = np.asarray(vectors)

    if len(vectors) == 0:
        raise ValueError("No neighbor vectors found.")

    lengths = np.linalg.norm(vectors, axis=1)

    # --------------------------------------------------
    # Estimate nearest-neighbor distance
    #
    # KD-tree nearest-neighbor distance is much safer
    # than median(all vectors), especially for hex lattice.
    # --------------------------------------------------
    dists, _ = tree.query(xy, k=2)

    nn_distance = np.median(dists[:, 1])

    # Keep only first coordination shell
    shell = (
        (lengths > 0.7 * nn_distance) &
        (lengths < 1.3 * nn_distance)
    )

    short_vectors = vectors[shell]

    if len(short_vectors) == 0:
        raise ValueError("Could not identify first neighbor shell.")

    # --------------------------------------------------
    # Convert vector orientation to angles.
    #
    # Opposite directions are equivalent:
    # theta and theta + pi represent same lattice axis.
    # --------------------------------------------------
    angles = np.arctan2(
        short_vectors[:, 1],
        short_vectors[:, 0]
    )

    angles = np.mod(angles, np.pi)

    # --------------------------------------------------
    # Find dominant first lattice direction
    # using circular histogram
    # --------------------------------------------------
    bins = 180

    hist, edges = np.histogram(
        angles,
        bins=bins,
        range=(0, np.pi)
    )

    theta_a = 0.5 * (
        edges[np.argmax(hist)] +
        edges[np.argmax(hist) + 1]
    )

    # Average vectors close to theta_a
    def angular_difference(theta1, theta2):
        """
        Difference between unoriented lattice directions.
        """
        d = np.abs(theta1 - theta2)
        return np.minimum(d, np.pi - d)

    tol = np.deg2rad(angle_tolerance_deg)

    mask_a = angular_difference(
        angles,
        theta_a
    ) < tol

    vecs_a = short_vectors[mask_a]

    # Orient consistently
    direction_a = np.array([
        np.cos(theta_a),
        np.sin(theta_a)
    ])

    vecs_a = np.array([
        v if np.dot(v, direction_a) > 0 else -v
        for v in vecs_a
    ])

    a = np.median(vecs_a, axis=0)

    # --------------------------------------------------
    # Second primitive direction should be ±60° from a
    # --------------------------------------------------
    theta_b1 = np.mod(theta_a + np.pi / 3, np.pi)
    theta_b2 = np.mod(theta_a - np.pi / 3, np.pi)

    diff1 = angular_difference(angles, theta_b1)
    diff2 = angular_difference(angles, theta_b2)

    if np.sum(diff1 < tol) >= np.sum(diff2 < tol):
        theta_b = theta_b1
        mask_b = diff1 < tol
    else:
        theta_b = theta_b2
        mask_b = diff2 < tol

    vecs_b = short_vectors[mask_b]

    direction_b = np.array([
        np.cos(theta_b),
        np.sin(theta_b)
    ])

    vecs_b = np.array([
        v if np.dot(v, direction_b) > 0 else -v
        for v in vecs_b
    ])

    b = np.median(vecs_b, axis=0)

    # --------------------------------------------------
    # Ensure a-b angle is ~60 rather than 120 degrees
    # --------------------------------------------------
    cos_angle = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if cos_angle < 0:
        b = -b

    return a, b

def generate_lattice_positions(xy, a, b, image_shape=None, margin=20):
    """
    Generate ideal lattice positions using origin p0 and lattice vectors a, b.

    Parameters
    ----------
    xy : ndarray, shape (N, 2)
        Reference points used to define the lattice origin.
    a, b : ndarray, shape (2,)
        Lattice basis vectors.
    image_shape : tuple or None
        Image shape as (height, width). If given, generated points are clipped
        to the image area.
    margin : float
        Extra margin around the coordinate bounds.

    Returns
    -------
    lattice_xy : ndarray, shape (K, 2)
        Generated ideal lattice positions.
    """

    xy = np.asarray(xy, dtype=float)

    # Use one reference atom as lattice origin
    p0 = xy[np.argmin(xy[:, 0] + xy[:, 1])]

    if image_shape is not None:
        height, width = image_shape
        xmin, xmax = -margin, width + margin
        ymin, ymax = -margin, height + margin
    else:
        xmin, ymin = xy.min(axis=0) - margin
        xmax, ymax = xy.max(axis=0) + margin

    lattice_points = []

    # Conservative index range
    max_range = 50

    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            p = p0 + i * a + j * b
            x, y = p

            if xmin <= x <= xmax and ymin <= y <= ymax:
                lattice_points.append(p)

    return np.array(lattice_points)

def classify_false_positives_by_lattice_symmetry(
    noisy_xy,
    false_positive_indices,
    lattice_xy,
    lattice_tolerance=3.0,
):
    """
    Classify false positives based on generated lattice positions.

    Bad FP:
        unmatched noisy Gaussian that sits close to a symmetry-derived
        lattice position.

    Good FP:
        unmatched noisy Gaussian that does not sit on the generated lattice.
    """

    noisy_xy = np.asarray(noisy_xy, dtype=float)
    lattice_xy = np.asarray(lattice_xy, dtype=float)

    false_positive_indices = list(false_positive_indices)

    good_fp = []
    bad_fp = []

    if len(false_positive_indices) == 0:
        return good_fp, bad_fp

    fp_xy = noisy_xy[false_positive_indices]

    tree = cKDTree(lattice_xy)

    nearest_distances, nearest_lattice_indices = tree.query(fp_xy, k=1)

    for fp_idx, nearest_dist, nearest_lattice_idx in zip(
        false_positive_indices,
        nearest_distances,
        nearest_lattice_indices,
    ):
        info = {
            "false_positive_index": int(fp_idx),
            "false_positive_xy": noisy_xy[fp_idx],
            "nearest_lattice_index": int(nearest_lattice_idx),
            "nearest_lattice_xy": lattice_xy[nearest_lattice_idx],
            "distance_to_lattice_position": float(nearest_dist),
        }

        if nearest_dist <= lattice_tolerance:
            bad_fp.append(info)
        else:
            good_fp.append(info)

    return good_fp, bad_fp


def compare_sublattices_with_lattice_fp_classification(
    reference_sublattice,
    noisy_sublattice,
    hexagonal = False,
    image_shape=None,
    match_tolerance=3.0,
    lattice_tolerance=3.0,
    neighbor_radius=25,
):
    """
    Compare reference and noisy Gaussian sublattices.

    False positives are classified using symmetry-derived lattice positions,
    not only the original reference points.
    """

    ref_xy = sublattice_to_xy(reference_sublattice)
    noisy_xy = sublattice_to_xy(noisy_sublattice)

    # Hungarian matching between reference and noisy detected positions
    cost = cdist(ref_xy, noisy_xy)
    ref_indices, noisy_indices = linear_sum_assignment(cost)

    matches = []
    false_negatives = set(range(len(ref_xy)))
    false_positives = set(range(len(noisy_xy)))

    for ref_idx, noisy_idx in zip(ref_indices, noisy_indices):
        d = cost[ref_idx, noisy_idx]

        if d <= match_tolerance:
            matches.append({
                "reference_index": int(ref_idx),
                "noisy_index": int(noisy_idx),
                "reference_atom": reference_sublattice.atom_list[ref_idx],
                "noisy_atom": noisy_sublattice.atom_list[noisy_idx],
                "reference_xy": ref_xy[ref_idx],
                "noisy_xy": noisy_xy[noisy_idx],
                "distance": float(d),
                "dx": float(noisy_xy[noisy_idx, 0] - ref_xy[ref_idx, 0]),
                "dy": float(noisy_xy[noisy_idx, 1] - ref_xy[ref_idx, 1]),
                "reference_sx": float(reference_sublattice.atom_list[ref_idx].sigma_x),
                "reference_sy": float(reference_sublattice.atom_list[ref_idx].sigma_y),
                "noisy_sx": float(noisy_sublattice.atom_list[noisy_idx].sigma_x),
                "noisy_sy": float(noisy_sublattice.atom_list[noisy_idx].sigma_y),
                "dsx": float(noisy_sublattice.atom_list[noisy_idx].sigma_x - reference_sublattice.atom_list[ref_idx].sigma_x),
                "dsy": float(noisy_sublattice.atom_list[noisy_idx].sigma_y - reference_sublattice.atom_list[ref_idx].sigma_y),
                "sigma_error": float(np.sqrt((noisy_sublattice.atom_list[noisy_idx].sigma_x - reference_sublattice.atom_list[ref_idx].sigma_x) ** 2 + (noisy_sublattice.atom_list[noisy_idx].sigma_y - reference_sublattice.atom_list[ref_idx].sigma_y) ** 2)),
                "amplitude error": float(noisy_sublattice.atom_list[noisy_idx].amplitude_gaussian - reference_sublattice.atom_list[ref_idx].amplitude_gaussian)/float(reference_sublattice.atom_list[ref_idx].amplitude_gaussian)
                })

            false_negatives.discard(ref_idx)
            false_positives.discard(noisy_idx)

    # Estimate lattice symmetry from reference positions  
    if hexagonal:
      a, b = estimate_hexagonal_lattice_vectors_from_points(
          ref_xy,
          neighbor_radius=neighbor_radius,
      )
    else:
      a, b = estimate_lattice_vectors_from_points(
          ref_xy,
          neighbor_radius=neighbor_radius,
      )

    # Generate ideal lattice positions
    lattice_xy = generate_lattice_positions(
        ref_xy,
        a,
        b,
        image_shape=image_shape,
        margin=20,
    )

    # Classify false positives using generated lattice sites
    good_fp, bad_fp = classify_false_positives_by_lattice_symmetry(
        noisy_xy=noisy_xy,
        false_positive_indices=false_positives,
        lattice_xy=lattice_xy,
        lattice_tolerance=lattice_tolerance,
    )

    errors = np.array([m["distance"] for m in matches])

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "matches": matches,
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),

        "good_false_positives": good_fp,
        "bad_false_positives": bad_fp,

        "num_true_positives": tp,
        "num_false_positives": fp,
        "num_false_negatives": fn,
        "num_good_false_positives": len(good_fp),
        "num_bad_false_positives": len(bad_fp),

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "mean_error": float(np.mean(errors)) if len(errors) > 0 else np.nan,
        "rmse_error": float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else np.nan,

        "lattice_vector_a": a,
        "lattice_vector_b": b,
        "generated_lattice_xy": lattice_xy,
        "sigma_error": float(np.mean(np.array([m["sigma_error"] for m in matches]))) if len(matches) > 0 else np.nan,
        "rmse_sigma_error": float(np.sqrt(np.mean(np.array([m["sigma_error"] for m in matches]) ** 2))) if len(matches) > 0  else np.nan,
        "amplitude_error": float(np.mean(np.array([m["amplitude error"] for m in matches]))) if len(matches) > 0 else np.nan,
        "rmse_amplitude_error": float(np.sqrt(np.mean(np.array([m["amplitude error"] for m in matches]) ** 2))) if len(matches) > 0 else np.nan
    }


# Helper function to extract coordinates from results for plotting
def get_classified_atom_coords(results, noisy_sublattice):
    noisy_xy = sublattice_to_xy(noisy_sublattice)

    tp_indices = [m['noisy_index'] for m in results['matches']]
    gfp_indices = [gfp_dict['false_positive_index'] for gfp_dict in results['good_false_positives']]
    bfp_indices = [bfp_dict['false_positive_index'] for bfp_dict in results['bad_false_positives']]

    # Modify to return empty 2D array with 0 rows and 2 columns when indices are empty
    tp_coords = noisy_xy[tp_indices] if tp_indices else np.empty((0, 2))
    gfp_coords = noisy_xy[gfp_indices] if gfp_indices else np.empty((0, 2))
    bfp_coords = noisy_xy[bfp_indices] if bfp_indices else np.empty((0, 2))

    return tp_coords, gfp_coords, bfp_coords

# Main plotting function
def plot_atom_classification(image, sublattice_clean, sublattice_noisy, results):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image.data, cmap='gray')
    ax.axis('off')

    # Plot reference atoms (clean) as 'x' markers
    #clean_xy = sublattice_to_xy(sublattice_clean)
    #ax.scatter(clean_xy[:, 0], clean_xy[:, 1], color='green', marker='x', s=100, label='Reference (Clean) Atoms')

    # Get classified noisy atom coordinates
    tp_coords, gfp_coords, bfp_coords = get_classified_atom_coords(results, sublattice_noisy)

    # Plot True Positives as green 'o' markers
    if tp_coords.shape[0] > 0:
        ax.scatter(tp_coords[:, 0], tp_coords[:, 1], color='green', marker='o', s=200, alpha=0.7, label='True Positives')

    # Plot Good False Positives as orange 'o' markers
    if gfp_coords.shape[0] > 0:
        ax.scatter(gfp_coords[:, 0], gfp_coords[:, 1], color='orange', marker='o', s=200, alpha=0.7, label='Good False Positives')

    # Plot Bad False Positives as red 'o' markers
    if bfp_coords.shape[0] > 0:
        ax.scatter(bfp_coords[:, 0], bfp_coords[:, 1], color='red', marker='o', s=200, alpha=0.7, label='Bad False Positives')

    #ax.set_title('Atom Classification on Clean Image')
    #ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()
