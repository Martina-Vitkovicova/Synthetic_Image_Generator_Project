from pathlib import Path
from typing import Optional, List

import numpy as np

from src.images_gen import generate_image, save_images
from utils.image_visualization import show_image_mask


def make_seeds(*, num: int, seed: Optional[int]) -> List[int]:
    """
    Create a list of root seeds.

    - If seed is None, generate `num` random seeds.
    - If seed is provided and num == 1, return [seed].
    - If seed is provided and num > 1, return a reproducible sequence derived from seed.
    """
    if num <= 0:
        return []

    if seed is None:
        rng = np.random.RandomState()
        return [int(rng.randint(0, 10**9)) for _ in range(num)]

    if num == 1:
        return [int(seed)]

    rng = np.random.RandomState(int(seed))
    return [int(rng.randint(0, 10**9)) for _ in range(num)]


def generate(num: int = 1, gen: str = "experiment1", img_params_path: str | Path = "config/base_params.json",
             img_save_path: str | Path = "data/gen", show: bool = True, save: bool = True, seed: int | None = 2):
    """
    Generate synthetic image-mask pairs using the configuration file in.

    Randomness is determined by `seed`. If `num > 1`, a reproducible list of per-image seeds is derived from `seed`.
    """
    img_save_path = Path(f"{img_save_path}_{gen}")
    seeds = make_seeds(num=num, seed=seed)

    for root_seed in seeds:
        img, mask = generate_image(root_seed, img_params_path)

        if show:
            show_image_mask(img, mask)
        if save:
            save_images(img_save_path, root_seed, img, mask, img_params_path, str(gen))


if __name__ == "__main__":
    generate(num=2,
             gen="experiment2",
             save=True)
