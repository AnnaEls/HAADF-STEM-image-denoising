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


def random_patch_mask_with_offset(
    x,
    patch_size=1,
    mask_ratio=0.2,
    offset=3,
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

    offset : int
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
        (-offset, -offset),
        (-offset, 0),
        (-offset, offset),

        (0, -offset),
        (0, offset),

        (offset, -offset),
        (offset, 0),
        (offset, offset),
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

