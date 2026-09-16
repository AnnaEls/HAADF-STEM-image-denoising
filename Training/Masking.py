import torch

def random_patch_mask(x, patch_size=1, mask_ratio=0.2, *, seed=None, epoch=None):
    """
    Random patch masking, reproducible across runs.
    """
    B, C, H, W = x.shape

    # Local deterministic RNG if seed given
    gen = None
    if seed is not None:
        s = int(seed) if epoch is None else int(seed) + int(epoch)
        gen = torch.Generator(device=x.device).manual_seed(s)

    mask = torch.ones((B, 1, H, W), device=x.device)

    num_patches = int(H * W * mask_ratio / (patch_size * patch_size))

    for _ in range(num_patches):
        top = torch.randint(0, H - patch_size, (1,), generator=gen, device=x.device).item()
        left = torch.randint(0, W - patch_size, (1,), generator=gen, device=x.device).item()
        mask[:, :, top:top+patch_size, left:left+patch_size] = 0

    return x * mask, mask


import torch


def random_patch_mask_with_dilation_old(
    x,
    patch_size=7,
    mask_ratio=0.2,
    dilation=3,
    *,
    seed=None,
    epoch=None
):
    """
    Random dilated blind-spot masking.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [B, C, H, W].

    patch_size : int
        Number of masked points along each dimension.
        Example: patch_size=7 -> 7 x 7 = 49 masked points
        per randomly selected dilated patch.

    mask_ratio : float
        Approximate fraction of pixels to mask.

    dilation : int
        Spacing between masked points.

        dilation=1:
            X X X X X X X

        dilation=2:
            X . X . X . X . X . X . X

        dilation=3:
            X . . X . . X . . X . . X . . X . . X

    seed : int or None
        Random seed.

    epoch : int or None
        Added to the seed to generate a different deterministic
        mask at each epoch.

    Returns
    -------
    masked_x : torch.Tensor
        Input with selected pixels set to zero.

    mask : torch.Tensor
        Binary mask [B, 1, H, W]:
            1 = visible/original pixel
            0 = masked pixel
    """

    B, C, H, W = x.shape

    # ----------------------------------------------------------
    # Random generator
    # ----------------------------------------------------------
    gen = None

    if seed is not None:
        s = int(seed) if epoch is None else int(seed) + int(epoch)

        gen = torch.Generator(device=x.device)
        gen.manual_seed(s)

    # ----------------------------------------------------------
    # Mask
    # ----------------------------------------------------------
    mask = torch.ones(
        (B, 1, H, W),
        dtype=x.dtype,
        device=x.device
    )

    # Effective spatial extent:
    #
    # patch_size = 7, dilation = 3
    #
    # X . . X . . X . . X . . X . . X . . X
    #
    # extent = 19 pixels
    # ----------------------------------------------------------
    effective_size = (
        1 + (patch_size - 1) * dilation
    )

    if effective_size > H or effective_size > W:
        raise ValueError(
            f"Effective dilated patch size is "
            f"{effective_size} x {effective_size}, "
            f"but image size is {H} x {W}."
        )

    # Number of actual masked pixels per patch
    n_masked_per_patch = patch_size * patch_size

    # Approximate number of random patch positions
    num_patches = max(
        1,
        int(
            H * W * mask_ratio /
            n_masked_per_patch
        )
    )

    # ----------------------------------------------------------
    # Generate random dilated patches
    # ----------------------------------------------------------
    for _ in range(num_patches):

        # Random top-left position of the whole effective region
        top = torch.randint(
            0,
            H - effective_size + 1,
            (1,),
            generator=gen,
            device=x.device
        ).item()

        left = torch.randint(
            0,
            W - effective_size + 1,
            (1,),
            generator=gen,
            device=x.device
        ).item()

        # ------------------------------------------------------
        # Coordinates of masked pixels
        #
        # For patch_size=7, dilation=3:
        #
        # top + [0, 3, 6, 9, 12, 15, 18]
        # ------------------------------------------------------
        ys = (
            top
            + torch.arange(
                patch_size,
                device=x.device
            ) * dilation
        )

        xs = (
            left
            + torch.arange(
                patch_size,
                device=x.device
            ) * dilation
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij"
        )

        # Set these pixels to masked
        mask[:, :, yy, xx] = 0

    # ----------------------------------------------------------
    # Genuine blind spot:
    # masked pixels are set to ZERO
    # ----------------------------------------------------------
    masked_x = x * mask

    return masked_x, mask


def random_patch_mask_with_dilation(
    x,
    patch_size=1,
    mask_ratio=0.2,
    dilation=3,
    *,
    seed=None,
    epoch=None
):
    """
    Random dilated blind patch masking, reproducible across runs.

    For each randomly selected patch, the patch is replaced by a patch
    sampled from a randomly chosen dilated neighbor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, C, H, W].

    patch_size : int
        Size of masked patch.

    mask_ratio : float
        Approximate fraction of pixels to mask.

    dilation : int
        Distance to the replacement patch.

    seed : int or None
        Random seed.

    epoch : int or None
        Added to seed so a different deterministic mask is generated
        for each epoch.

    Returns
    -------
    masked_x : torch.Tensor
        Input with masked regions replaced by dilated neighboring patches.

    mask : torch.Tensor
        Binary mask of shape [B, 1, H, W]:
            1 = original/visible
            0 = masked/replaced
    """

    B, C, H, W = x.shape

    # ------------------------------------------------------------
    # Local deterministic RNG
    # ------------------------------------------------------------
    gen = None

    if seed is not None:
        s = int(seed) if epoch is None else int(seed) + int(epoch)
        gen = torch.Generator(device=x.device).manual_seed(s)

    # 1 = visible
    # 0 = masked
    mask = torch.ones(
        (B, 1, H, W),
        device=x.device,
        dtype=x.dtype
    )

    masked_x = x.clone()

    num_patches = int(
        H * W * mask_ratio /
        (patch_size * patch_size)
    )

    # ------------------------------------------------------------
    # Dilated neighbors
    # ------------------------------------------------------------
    offsets = [
        (-dilation, -dilation),
        (-dilation, 0),
        (-dilation, dilation),

        (0, -dilation),
        (0, dilation),

        (dilation, -dilation),
        (dilation, 0),
        (dilation, dilation),
    ]

    # ------------------------------------------------------------
    # Mask patches
    # ------------------------------------------------------------
    for _ in range(num_patches):

        top = torch.randint(
            0,
            H - patch_size + 1,
            (1,),
            generator=gen,
            device=x.device
        ).item()

        left = torch.randint(
            0,
            W - patch_size + 1,
            (1,),
            generator=gen,
            device=x.device
        ).item()

        # randomly choose one dilated neighbor
        idx = torch.randint(
            0,
            len(offsets),
            (1,),
            generator=gen,
            device=x.device
        ).item()

        dy, dx = offsets[idx]

        # --------------------------------------------------------
        # Replacement coordinates
        # --------------------------------------------------------
        src_top = top + dy
        src_left = left + dx

        # Keep source patch inside the image
        src_top = max(
            0,
            min(src_top, H - patch_size)
        )

        src_left = max(
            0,
            min(src_left, W - patch_size)
        )

        # --------------------------------------------------------
        # Replace selected patch
        # --------------------------------------------------------
        masked_x[
            :,
            :,
            top:top + patch_size,
            left:left + patch_size
        ] = x[
            :,
            :,
            src_top:src_top + patch_size,
            src_left:src_left + patch_size
        ]

        # mark as masked
        mask[
            :,
            :,
            top:top + patch_size,
            left:left + patch_size
        ] = 0

    return masked_x, mask

