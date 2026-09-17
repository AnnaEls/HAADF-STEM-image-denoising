"""
correlated_abtem_noise.py

Spatially correlated Poisson-derived noise for abTEM measurements.

This module leaves the installed abTEM package unchanged.

Typical use
-----------
from correlated_abtem_noise import correlated_poisson_noise

noisy = correlated_poisson_noise(
    measurement,
    dose_per_area=100,
    seed=42,
    correlation_sigma=(0.5, 2.0),
)

Optional monkey patch
---------------------
from correlated_abtem_noise import patch_poisson_noise

patch_poisson_noise(measurement)

noisy = measurement.poisson_noise(
    dose_per_area=100,
    seed=42,
    correlation_sigma=2.0,
)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from abtem.core.backend import get_array_module
from abtem.core.utils import get_dtype
from abtem.distributions import BaseDistribution
from abtem.noise import NoiseTransform


class CorrelatedNoiseTransform(NoiseTransform):
    """
    abTEM Poisson-noise transform with optional spatial correlation.

    The standard abTEM Poisson realization is first generated in count space:

        N ~ Poisson(lambda)

    where lambda is the expected count image.

    The Poisson residual

        r = N - lambda

    is then spatially filtered:

        r_corr = G * r

    and the final result is

        N_corr = lambda + r_corr

    Thus, correlation is introduced into the noise residual while the expected
    signal is retained.

    Notes
    -----
    After spatial filtering, the output is no longer strictly Poisson-distributed
    at each pixel. It is best described as spatially correlated, Poisson-derived
    noise.

    Parameters
    ----------
    dose
        Same meaning as in abTEM's NoiseTransform.
    samples
        Number of noisy realizations.
    seeds
        Random seed or seeds.
    correlation_sigma
        Gaussian correlation length in pixels.

        * 0.0 -> standard independent Poisson noise
        * 1.5 -> isotropic correlation
        * (0.5, 2.0) -> anisotropic correlation, sigma_y=0.5, sigma_x=2.0

    preserve_variance
        If True, restore the spatial standard deviation of the original Poisson
        residual after Gaussian filtering.

    boundary
        Boundary condition passed to scipy.ndimage.gaussian_filter.
        Common choices are "reflect", "wrap", and "nearest".

    clip_nonnegative
        If True, clip the final count image to >= 0.
    """

    def __init__(
        self,
        dose: float | np.ndarray | BaseDistribution,
        samples: Optional[int] = None,
        seeds: Optional[int | tuple[int, ...]] = None,
        correlation_sigma: float | tuple[float, float] = 0.0,
        preserve_variance: bool = True,
        boundary: str = "reflect",
        clip_nonnegative: bool = True,
    ):
        super().__init__(
            dose=dose,
            samples=samples,
            seeds=seeds,
        )

        self._correlation_sigma = correlation_sigma
        self._preserve_variance = bool(preserve_variance)
        self._boundary = boundary
        self._clip_nonnegative = bool(clip_nonnegative)

    @property
    def correlation_sigma(self) -> float | tuple[float, float]:
        return self._correlation_sigma

    @property
    def preserve_variance(self) -> bool:
        return self._preserve_variance

    @property
    def boundary(self) -> str:
        return self._boundary

    @property
    def clip_nonnegative(self) -> bool:
        return self._clip_nonnegative

    def _calculate_new_array(self, array_object):
        array = array_object._eager_array
        xp = get_array_module(array)

        # ------------------------------------------------------------
        # Reproduce abTEM's original dose/sample handling
        # ------------------------------------------------------------
        if isinstance(self.seeds, BaseDistribution):
            array = xp.tile(
                array[None],
                (self.samples,) + (1,) * len(array.shape),
            )

        if isinstance(self.dose, BaseDistribution):
            dose = xp.array(
                self.dose.values,
                dtype=get_dtype(),
            )

            array = array[None] * xp.expand_dims(
                dose,
                tuple(range(1, len(array.shape) + 1)),
            )
        else:
            array = array * xp.asarray(
                self.dose,
                dtype=get_dtype(),
            )

        # ------------------------------------------------------------
        # Reproduce abTEM's RNG logic
        # ------------------------------------------------------------
        if isinstance(self.seeds, BaseDistribution):
            seed = sum(self.seeds.values)
        else:
            seed = self.seeds

        seed_rng = np.random.default_rng(seed=seed)

        randomized_seed = int(
            seed_rng.integers(np.iinfo(np.int32).max)
        )

        poisson_rng = np.random.RandomState(
            seed=randomized_seed
        )

        # ------------------------------------------------------------
        # Poisson sampling on CPU, as in abTEM
        # ------------------------------------------------------------
        expected_counts = (
            array.get()
            if hasattr(array, "get")
            else np.asarray(array)
        )

        expected_counts = np.clip(
            expected_counts,
            a_min=0.0,
            a_max=None,
        )

        poisson_counts = poisson_rng.poisson(
            expected_counts
        ).astype(get_dtype())

        # sigma = 0 -> exactly the original abTEM behavior
        sigma = self.correlation_sigma

        if not np.any(np.asarray(sigma, dtype=float) > 0):
            return xp.asarray(poisson_counts)

        # ------------------------------------------------------------
        # Poisson residual
        # ------------------------------------------------------------
        noise = (
            poisson_counts.astype(np.float64)
            - expected_counts.astype(np.float64)
        )

        if noise.ndim < 2:
            raise ValueError(
                "CorrelatedNoiseTransform requires at least two spatial "
                "dimensions."
            )

        # ------------------------------------------------------------
        # Correlate only the last two dimensions (..., y, x).
        #
        # This avoids mixing dose/sample/ensemble axes.
        # ------------------------------------------------------------
        if np.isscalar(sigma):
            sigma_y = float(sigma)
            sigma_x = float(sigma)
        else:
            if len(sigma) != 2:
                raise ValueError(
                    "correlation_sigma must be a scalar or "
                    "(sigma_y, sigma_x)."
                )

            sigma_y = float(sigma[0])
            sigma_x = float(sigma[1])

        if sigma_y < 0 or sigma_x < 0:
            raise ValueError(
                "correlation_sigma values must be >= 0."
            )

        filter_sigma = (
            (0.0,) * (noise.ndim - 2)
            + (sigma_y, sigma_x)
        )

        correlated_noise = gaussian_filter(
            noise,
            sigma=filter_sigma,
            mode=self.boundary,
        )

        # ------------------------------------------------------------
        # Preserve the finite-image residual mean and, optionally,
        # its spatial standard deviation.
        # ------------------------------------------------------------
        spatial_axes = (-2, -1)

        mean_original = np.mean(
            noise,
            axis=spatial_axes,
            keepdims=True,
        )

        mean_correlated = np.mean(
            correlated_noise,
            axis=spatial_axes,
            keepdims=True,
        )

        correlated_centered = (
            correlated_noise - mean_correlated
        )

        if self.preserve_variance:
            std_original = np.std(
                noise,
                axis=spatial_axes,
                keepdims=True,
            )

            std_correlated = np.std(
                correlated_centered,
                axis=spatial_axes,
                keepdims=True,
            )

            scale = np.divide(
                std_original,
                std_correlated,
                out=np.ones_like(std_original),
                where=std_correlated > 1e-12,
            )

            correlated_centered *= scale

        correlated_noise = (
            correlated_centered + mean_original
        )

        # ------------------------------------------------------------
        # Add correlated residual back to the expected count image
        # ------------------------------------------------------------
        output = (
            expected_counts.astype(np.float64)
            + correlated_noise
        )

        if self.clip_nonnegative:
            output = np.clip(
                output,
                a_min=0.0,
                a_max=None,
            )

        return xp.asarray(
            output.astype(get_dtype())
        )


def correlated_poisson_noise(
    measurement,
    dose_per_area: float | Sequence[float] | None = None,
    total_dose: float | Sequence[float] | None = None,
    samples: int = 1,
    seed: int | None = None,
    correlation_sigma: float | tuple[float, float] = 0.0,
    preserve_variance: bool = True,
    boundary: str = "reflect",
    clip_nonnegative: bool = True,
):
    """
    Add spatially correlated Poisson-derived noise to an abTEM measurement.

    Parameters
    ----------
    measurement
        abTEM measurement object.

    dose_per_area
        Dose in electrons / Å^2. The measurement must provide
        ``_area_per_pixel``.

    total_dose
        Total dose. Supply exactly one of ``dose_per_area`` and
        ``total_dose``.

    samples
        Number of noisy realizations.

    seed
        Random seed.

    correlation_sigma
        Spatial Gaussian correlation width in pixels.

        Examples
        --------
        0.0
            Standard independent Poisson noise.

        1.5
            Isotropic correlation.

        (0.5, 2.0)
            Anisotropic correlation:
            sigma_y = 0.5, sigma_x = 2.0.

    preserve_variance
        Restore the variance lost by Gaussian filtering.

    boundary
        Boundary mode for Gaussian filtering.

    clip_nonnegative
        Clip the final count array to non-negative values.

    Returns
    -------
    abTEM measurement
        Noisy measurement.
    """

    if (
        dose_per_area is not None
        and total_dose is not None
    ):
        raise RuntimeError(
            "Provide one of 'dose_per_area' or 'total_dose', not both."
        )

    if (
        dose_per_area is None
        and total_dose is None
    ):
        raise RuntimeError(
            "Provide one of 'dose_per_area' or 'total_dose'."
        )

    dtype = get_dtype(complex=False)

    if dose_per_area is not None:
        dose_per_area_array = np.asarray(
            dose_per_area,
            dtype=dtype,
        )

        total_dose = (
            measurement._area_per_pixel
            * dose_per_area_array
        )

    total_dose = np.asarray(
        total_dose,
        dtype=dtype,
    )

    transform = CorrelatedNoiseTransform(
        dose=total_dose,
        samples=samples,
        seeds=seed,
        correlation_sigma=correlation_sigma,
        preserve_variance=preserve_variance,
        boundary=boundary,
        clip_nonnegative=clip_nonnegative,
    )

    # Current abTEM NoiseTransform.apply() uses array_object.apply_transform().
    return transform.apply(measurement)


def _patched_poisson_noise(
    self,
    dose_per_area: float | Sequence[float] | None = None,
    total_dose: float | Sequence[float] | None = None,
    samples: int = 1,
    seed: int | None = None,
    correlation_sigma: float | tuple[float, float] = 0.0,
    preserve_variance: bool = True,
    boundary: str = "reflect",
    clip_nonnegative: bool = True,
):
    """
    Replacement ``poisson_noise`` method used by ``patch_poisson_noise``.
    """

    return correlated_poisson_noise(
        self,
        dose_per_area=dose_per_area,
        total_dose=total_dose,
        samples=samples,
        seed=seed,
        correlation_sigma=correlation_sigma,
        preserve_variance=preserve_variance,
        boundary=boundary,
        clip_nonnegative=clip_nonnegative,
    )


def patch_poisson_noise(measurement_or_class):
    """
    Replace ``poisson_noise`` on a measurement class for the current Python
    session.

    Parameters
    ----------
    measurement_or_class
        Either an abTEM measurement instance or its class.

    Returns
    -------
    type
        The patched class.

    Example
    -------
    patch_poisson_noise(measurement)

    noisy = measurement.poisson_noise(
        dose_per_area=100,
        seed=42,
        correlation_sigma=(0.5, 2.0),
    )
    """

    if isinstance(measurement_or_class, type):
        cls = measurement_or_class
    else:
        cls = type(measurement_or_class)

    cls.poisson_noise = _patched_poisson_noise

    return cls


__all__ = [
    "CorrelatedNoiseTransform",
    "correlated_poisson_noise",
    "patch_poisson_noise",
]
