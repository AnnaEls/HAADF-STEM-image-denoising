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

def keep_lattice_inside_image(lattice_xy, image, margin=10):
    """
    Keep only lattice points safely inside the image.

    Parameters
    ----------
    lattice_xy : ndarray, shape (N, 2)
        Lattice positions [x, y].

    image : HyperSpy Signal2D or ndarray
        Image used for bounds.

    margin : float
        Number of pixels excluded from each image edge.
    """

    if hasattr(image, "data"):
        img = image.data
    else:
        img = image

    H, W = img.shape

    x = lattice_xy[:, 0]
    y = lattice_xy[:, 1]

    mask = (
        (x >= margin) &
        (x < W - margin) &
        (y >= margin) &
        (y < H - margin)
    )

    return lattice_xy[mask]

def find_vacancies_from_lattice(
    sublattice,
    lattice_xy,
    vacancy_tolerance=4.0
):
    """
    Find expected lattice positions that do not contain
    a detected atom.

    Parameters
    ----------
    clean_sublattice : Atomap Sublattice
        Detected atoms in clean/reference image.

    lattice_xy : ndarray, shape (N, 2)
        Expected ideal lattice positions.

    vacancy_tolerance : float
        Maximum distance [pixels] for considering a lattice
        position occupied.

    Returns
    -------
    vacancy_xy : ndarray
        Coordinates of missing lattice sites.
    occupied_xy : ndarray
        Coordinates of occupied lattice sites.
    """

    atom_xy = sublattice_to_xy(sublattice)

    tree = cKDTree(atom_xy)

    distances, nearest_atom = tree.query(
        lattice_xy,
        k=1
    )

    occupied_mask = distances <= vacancy_tolerance

    occupied_xy = lattice_xy[occupied_mask]
    vacancy_xy = lattice_xy[~occupied_mask]

    return vacancy_xy, occupied_xy, distances

def plot_vacancies(
    image,
    clean_sublattice,
    vacancy_xy,
    s_v = 250,
    s_a = 35,
    show_atoms=True
):
    if hasattr(image, "data"):
        img = image.data
    else:
        img = image

    atom_xy = sublattice_to_xy(clean_sublattice)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img, cmap="gray")

    if show_atoms:
        ax.scatter(
            atom_xy[:, 0],
            atom_xy[:, 1],
            marker="x",
            s=35,
            c="lime",
            linewidths=1.3,
            label="Detected atoms"
        )

    if len(vacancy_xy) > 0:
        ax.scatter(
            vacancy_xy[:, 0],
            vacancy_xy[:, 1],
            s=s_V,
            facecolors="none",
            edgecolors="lime",
            linewidths=2.5,
            label="Vacancy"
        )

    ax.axis("off")
    ax.legend()

    plt.tight_layout()
    plt.show()

def match_reference_and_noisy(
    reference_sublattice,
    noisy_sublattice,
    match_tolerance=5.0
):
    """
    Match noisy detections directly to nearest reference atoms.

    A noisy detection is:
        TP if a reference atom exists within match_tolerance
        FP otherwise

    A reference atom is:
        FN if no noisy detection was matched to it
    """

    ref_xy = sublattice_to_xy(reference_sublattice)
    noisy_xy = sublattice_to_xy(noisy_sublattice)

    # --------------------------------------------------
    # Nearest reference atom for every noisy atom
    # --------------------------------------------------
    tree = cKDTree(ref_xy)

    distances, nearest_ref = tree.query(
        noisy_xy,
        k=1
    )

    # Candidate matches inside tolerance
    candidates = np.where(
        distances <= match_tolerance
    )[0]

    # --------------------------------------------------
    # Handle duplicate noisy detections:
    #
    # Several noisy points could point to the same
    # reference atom. Keep only the closest one.
    # --------------------------------------------------
    candidates_by_ref = {}

    for noisy_idx in candidates:

        ref_idx = int(nearest_ref[noisy_idx])
        d = float(distances[noisy_idx])

        if ref_idx not in candidates_by_ref:
            candidates_by_ref[ref_idx] = []

        candidates_by_ref[ref_idx].append(
            (noisy_idx, d)
        )

    matches = []

    matched_noisy = set()
    matched_ref = set()

    for ref_idx, candidate_list in candidates_by_ref.items():

        # closest noisy detection wins
        noisy_idx, d = min(
            candidate_list,
            key=lambda x: x[1]
        )

        matches.append({
            "reference_index": int(ref_idx),
            "noisy_index": int(noisy_idx),

            "reference_xy":
                ref_xy[ref_idx],

            "noisy_xy":
                noisy_xy[noisy_idx],

            "distance":
                float(d)
        })

        matched_ref.add(ref_idx)
        matched_noisy.add(noisy_idx)

    # --------------------------------------------------
    # Everything else in noisy image = FP
    # --------------------------------------------------
    false_positive_indices = sorted(
        set(range(len(noisy_xy))) -
        matched_noisy
    )

    # Reference atoms without detections = FN
    false_negative_indices = sorted(
        set(range(len(ref_xy))) -
        matched_ref
    )

    return {

        "matches": matches,

        "false_positive_indices":
            false_positive_indices,

        "false_positive_xy":
            noisy_xy[false_positive_indices],

        "false_negative_indices":
            false_negative_indices,

        "false_negative_xy":
            ref_xy[false_negative_indices],

        "num_true_positives":
            len(matches),

        "num_false_positives":
            len(false_positive_indices),

        "num_false_negatives":
            len(false_negative_indices),

        
        "nearest_distances": distances,
        "nearest_reference_indices": nearest_ref
    }

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
