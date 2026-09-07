import numpy as np
from scipy.ndimage import maximum_filter


def estimate_real_lattice_vectors_fft(
    image,
    pixel_size=1.0,
    dc_radius=5,
    min_peak_distance=4,
    num_peaks=30,
    target_angle=60,
    angle_tolerance=15,
    apply_hann=True
):
    """
    Estimate real-space lattice vectors from FFT peaks.

    Parameters
    ----------
    image : (H, W) ndarray
        Input lattice image.

    pixel_size : float
        Real-space size per pixel.
        Use 1.0 to obtain lattice vectors in pixels.
        Example: 0.01 for nm/pixel.

    Returns
    -------
    a1, a2 : ndarray, shape (2,)
        Real-space lattice vectors [x, y].

        If pixel_size=1:
            units = pixels

        If pixel_size=0.01:
            units = nm
    """

    image = np.asarray(image, dtype=float)
    H, W = image.shape

    
    if apply_hann:
        window = np.outer(np.hanning(H), np.hanning(W))
        img = image * window
    else:
        img = image

   
    F = np.fft.fftshift(np.fft.fft2(img))
    spectrum = np.log1p(np.abs(F))

    cy, cx = H // 2, W // 2

    yy, xx = np.indices((H, W))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    spectrum = spectrum.copy()
    spectrum[rr < dc_radius] = 0

    
    size = 2 * min_peak_distance + 1

    maxima = spectrum == maximum_filter(
        spectrum,
        size=size
    )

    py, px = np.where(maxima)

    strength = spectrum[py, px]
    order = np.argsort(strength)[::-1]

    px = px[order][:num_peaks]
    py = py[order][:num_peaks]

    # relative FFT coordinates [kx, ky]
    peaks = np.column_stack((
        px - cx,
        py - cy
    )).astype(float)

   
    unique = []

    for p in peaks:

        duplicate = False

        for q in unique:
            if (
                np.linalg.norm(p - q) < min_peak_distance
                or
                np.linalg.norm(p + q) < min_peak_distance
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    unique = np.asarray(unique)

   
    best_pair = None
    best_score = np.inf

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):

            p1 = unique[i]
            p2 = unique[j]

            r1 = np.linalg.norm(p1)
            r2 = np.linalg.norm(p2)

            cosang = np.dot(p1, p2) / (r1 * r2)
            cosang = np.clip(cosang, -1, 1)

            angle = np.degrees(
                np.arccos(abs(cosang))
            )

            angle_error = abs(angle - target_angle)

            if angle_error > angle_tolerance:
                continue

            # require similar reciprocal lengths
            radius_error = abs(r1 - r2) / ((r1 + r2) / 2)

            score = (
                angle_error
                + 20 * radius_error
                + 0.01 * (r1 + r2)
            )

            if score < best_score:
                best_score = score
                best_pair = (p1, p2)

    if best_pair is None:
        raise RuntimeError(
            "Could not identify two lattice FFT peaks."
        )

    p1, p2 = best_pair

    
    g1 = np.array([
        p1[0] / (W * pixel_size),
        p1[1] / (H * pixel_size)
    ])

    g2 = np.array([
        p2[0] / (W * pixel_size),
        p2[1] / (H * pixel_size)
    ])

    G = np.array([
        g1,
        g2
    ])

    A = np.linalg.inv(G)

    a1 = A[:, 0]
    a2 = A[:, 1]

    return a1, a2
