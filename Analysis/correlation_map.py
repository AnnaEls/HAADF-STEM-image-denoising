import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

def pearson_correlation_map(image_1, image_2, a1, a2):

  image_1_data = np.array(image_1)
  image_2_data = np.array(image_2)
  A = image_1_data.astype(np.float64)
  B = image_2_data.astype(np.float64)
  
  if A.shape != B.shape:
    raise ValueError(f"Different image shapes: {A.shape} vs {B.shape}")
    
  H, W = A.shape

  # Average image for lattice estimation
  I = 0.5*(A+B)
  I_smooth = gaussian_filter(I, sigma=1.0)

  #Detect visible atoms only to estimate the lattice
  detected_yx = peak_local_max(
    I_smooth,
    min_distance=5,
    threshold_abs=np.percentile(I_smooth, 50),
    exclude_border=5
  )

  # convert [y,x] -> [x,y]
  detected_xy = detected_yx[:, ::-1].astype(float)


  A_lattice = np.column_stack([a1, a2])


  # Estimate lattice origin / phase
  A_inv = np.linalg.inv(A_lattice)

  # Convert detected atom positions into lattice coordinates
  uv = (A_inv @ detected_xy.T).T

  # fractional coordinates
  frac = uv - np.floor(uv)

  phase = []

  for j in range(2):
      z = np.mean(np.exp(2j * np.pi * frac[:, j]))
      p = np.angle(z) / (2 * np.pi)
      if p < 0:
        p += 1
      phase.append(p)

  phase = np.array(phase)


  # Generate ALL expected lattice sites
  expected_sites = []
  lattice_indices = []

  margin = 5

  for i in range(-100, 100):
      for j in range(-100, 100):
        uv_site = np.array([i, j], dtype=float) + phase
        xy = A_lattice @ uv_site
        x, y = xy
        if (
            margin <= x < W - margin
            and margin <= y < H - margin
        ):
            expected_sites.append([x, y])
            lattice_indices.append([i, j])


  expected_sites = np.asarray(expected_sites)
  lattice_indices = np.asarray(lattice_indices)

  # Refinement of the global lattice translation
  tree_detected = cKDTree(detected_xy)

  dist, nearest = tree_detected.query(expected_sites)

  # use only good matches
  good = dist < 4.0

  offsets = detected_xy[nearest[good]] - expected_sites[good]

  global_shift = np.median(offsets, axis=0)

  expected_sites += global_shift

  # Create one cell around every expected atom
  yy, xx = np.indices((H, W))
  pixel_xy = np.column_stack([
     xx.ravel(),
     yy.ravel()
  ])

  tree_lattice = cKDTree(expected_sites)

  distance_to_site, cell_index = tree_lattice.query(pixel_xy)

  cell_map = cell_index.reshape(H, W)

  #Pearson correlation for every cell
                 
  n_cells = len(expected_sites)

  cell_corr = np.full(n_cells, np.nan)
  cell_area = np.zeros(n_cells, dtype=int)

  cell_mean_afno = np.full(n_cells, np.nan)
  cell_mean_cnn  = np.full(n_cells, np.nan)

  for k in range(n_cells):
    mask = cell_map == k

    afno_cell = A[mask]
    cnn_cell  = B[mask]

    cell_area[k] = mask.sum()

    cell_mean_afno[k] = np.mean(afno_cell)
    cell_mean_cnn[k]  = np.mean(cnn_cell)

    if (
        len(afno_cell) >= 4
        and np.std(afno_cell) > 1e-12
        and np.std(cnn_cell) > 1e-12
    ):
        cell_corr[k] = np.corrcoef(
            afno_cell,
            cnn_cell
        )[0, 1]

  return cell_corr, cell_map, cell_area, expected_sites
