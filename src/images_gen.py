import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.background_gen import generate_matrix
from src.etch_pits_gen import add_etch_pits
from src.precipitates_gen import add_precipitates
from utils.type_definitions import Image, Mask


def generate_image(root_seed: int, img_params_path: str) -> Tuple[Image, Mask]:
    """
    Generate a single synthetic image and precipitate mask from a config file.

    The pipeline applies matrix generation, precipitates and noise terms. Randomness is fully determined by `root_seed`.
    """
    rng = np.random.RandomState(root_seed)
    vor_seed = rng.randint(0, 10 ** 5)
    precip_seed = rng.randint(0, 10 ** 5)
    noise_seed = rng.randint(0, 10 ** 5)
    etch_pits_seed = rng.randint(0, 10 ** 5)

    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    background = generate_matrix(vor_seed, **img_params)
    image = add_etch_pits(background, etch_pits_seed, **img_params)
    image, mask = add_precipitates(image, precip_seed, **img_params)

    image = add_gaussian_noise(image, noise_seed)
    image = np.clip(image, 0.0, 255.0)

    return image, mask


def save_images(root_path: Path, seed: int, img: Image, mask: Mask, img_params_path: str | None, gen: str) -> None:
    root_path = Path(root_path)
    path = root_path / f"seed-{seed}"
    path.mkdir(exist_ok=True, parents=True)
    plt.imsave(path / "img.png", img, cmap='gray', vmin=0, vmax=255)
    plt.imsave(path / "mask.png", mask, cmap='gray', vmin=0, vmax=255)

    assert os.path.isfile(path / "img.png") and os.path.isfile(path / "mask.png")

    if img_params_path:
        save_metadata(img_params_path, seed, gen, path)


def save_metadata(img_params_path: str | Path, seed: int, gen: str, path: str | Path):
    with open(img_params_path, "r") as file:
        img_params = json.load(file)

    updated_img_params = img_params.copy()
    updated_img_params["seed"] = seed
    updated_img_params["gen"] = gen

    with open(path / "metadata.json", 'w') as file:
        json.dump(updated_img_params, file, indent=4)


def add_gaussian_noise(image: Image, seed: int) -> Image:
    h, w = image.shape
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise -= float(noise.mean())

    alpha = float(rng.randint(2, 4)) * 0.01
    return image + noise * (alpha * 127.5)
