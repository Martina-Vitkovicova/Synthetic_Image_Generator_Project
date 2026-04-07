import pathlib
import numpy as np

from src.images_gen import generate_image, save_images
from utils.image_visualization import show_image_mask

GENERATION = 10
IMG_PARAMS_PATH = "config/base_params.json"
IMG_SAVE_PATH = pathlib.Path("data/gen" + str(GENERATION))

for _ in range(1):
    root_seed = np.random.randint(0, 10 ** 5)
    img, mask = generate_image(root_seed, IMG_PARAMS_PATH)

    show_image_mask(img, mask)

    save_images(IMG_SAVE_PATH, root_seed, img, mask, IMG_PARAMS_PATH, str(GENERATION))
