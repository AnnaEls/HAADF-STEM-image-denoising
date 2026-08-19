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

def random_validation_mask(x, val_ratio=0.05, seed=1234):
    """
    Fixed random validation target mask.

    Returns
    -------
    val_mask : bool tensor [B, 1, H, W]
        True  = reserved validation pixel
        False = available for training
    """
    B, C, H, W = x.shape

    gen = torch.Generator(device=x.device).manual_seed(seed)

    val_mask = torch.zeros(
        (B, 1, H, W),
        dtype=torch.bool,
        device=x.device
    )

    n_val = int(H * W * val_ratio)

    for b in range(B):
        idx = torch.randperm(
            H * W,
            generator=gen,
            device=x.device
        )[:n_val]

        val_mask[b, 0].view(-1)[idx] = True

    return val_mask

def random_patch_mask_with_validation(
    x,
    patch_size=1,
    mask_ratio=0.2,
    *,
    seed=None,
    epoch=None,
    val_mask=None,
):
    """
    Random masking for self-supervised training.

    Validation pixels are excluded from training targets.

    Parameters
    ----------
    x : tensor
        [B, C, H, W]

    patch_size : int
        Size of masked patch.

    mask_ratio : float
        Fraction of available training pixels to mask.

    seed : int or None
        Base random seed.

    epoch : int or None
        Added to seed so training mask changes every epoch.

    val_mask : bool tensor or None
        [B, 1, H, W]
        True = reserved validation pixel.

    Returns
    -------
    x_masked
        Masked input.

    mask
        1 = visible
        0 = masked training target
    """

    B, C, H, W = x.shape

    # -----------------------------------
    # RNG: deterministic but epoch-varying
    # -----------------------------------
    gen = None

    if seed is not None:
        s = int(seed)

        if epoch is not None:
            s += int(epoch)

        gen = torch.Generator(
            device=x.device
        ).manual_seed(s)

    # 1 = visible
    # 0 = masked
    mask = torch.ones(
        (B, 1, H, W),
        device=x.device
    )

    # -----------------------------------
    # Pixels that may be training targets
    # -----------------------------------
    if val_mask is None:
        allowed = torch.ones(
            (B, 1, H, W),
            dtype=torch.bool,
            device=x.device
        )
    else:
        allowed = ~val_mask.bool()

    # -----------------------------------
    # Pixel masking
    # -----------------------------------
    if patch_size == 1:

        for b in range(B):

            allowed_idx = torch.where(
                allowed[b, 0].flatten()
            )[0]

            n_mask = int(
                len(allowed_idx) * mask_ratio
            )

            perm = torch.randperm(
                len(allowed_idx),
                generator=gen,
                device=x.device
            )

            selected = allowed_idx[
                perm[:n_mask]
            ]

            mask[b, 0].view(-1)[selected] = 0

    # -----------------------------------
    # Patch masking
    # -----------------------------------
    else:

        num_patches = int(
            H * W * mask_ratio /
            (patch_size * patch_size)
        )

        for b in range(B):

            count = 0
            attempts = 0

            max_attempts = num_patches * 100

            while (
                count < num_patches
                and attempts < max_attempts
            ):

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

                region_allowed = allowed[
                    b,
                    0,
                    top:top + patch_size,
                    left:left + patch_size
                ]

                # Do not allow training patches
                # to contain validation pixels
                if region_allowed.all():

                    mask[
                        b,
                        0,
                        top:top + patch_size,
                        left:left + patch_size
                    ] = 0

                    count += 1

                attempts += 1

    return x * mask, mask
