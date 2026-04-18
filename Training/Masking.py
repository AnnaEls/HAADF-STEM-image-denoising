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
