from typing import List, Tuple

import cv2
import numpy as np

from utils.type_definitions import Image, Point, RandomState, Contour
from utils.light_normalisation import normalise_light_direction


def add_etch_pits(image: Image, seed: int, *, etch_pits_num_range: List[int],
                  etch_pits_size_range: List[int], etch_pits_color_range: List[int],
                  light_direction: List[int], etch_pits_shading_values: List[int], **kwargs) -> Image:
    """
    Add synthetic etch pits to a grayscale image.

    Parameters
    ----------
    image : Image
        Input grayscale image.
    seed : int
        Random seed used for reproducible pit generation.
    etch_pits_num_range : list[int]
        Range controlling how many etch pits are generated.
    etch_pits_size_range : list[int]
        Range controlling pit size.
    etch_pits_color_range : list[int]
        Range of base grayscale intensities used for pits.
    light_direction : list[int]
        Three-component light direction vector used to for shading.
    etch_pits_shading_values : list[int]
        Intensity values for the pit shading components in the order
        [dark_shadow, light_shadow, highlight].

    Returns
    -------
    Image with generated etch pits.
    """
    rng = np.random.RandomState(seed)
    num = rng.randint(*etch_pits_num_range)
    dark_shadow, light_shadow, light = etch_pits_shading_values
    h, w = image.shape

    # Padding prevents pits near image boundaries from producing edge artifacts.
    pad = int(max(etch_pits_size_range))
    pad_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    pad_h, pad_w = pad_image.shape

    for _ in range(num):
        angle = rng.randint(0, 180)
        size = _generate_pit_size(rng, etch_pits_size_range)
        base_color = rng.randint(*etch_pits_color_range)
        centers = generate_centers(rng, pad_w, pad_h, light_direction, size)
        colors = [light, dark_shadow, light_shadow, base_color]

        pad_image = add_colored_shapes(pad_image, rng, centers, size, angle, colors, light_direction)

    # crop back to original size
    image = pad_image[pad:pad + h, pad:pad + w]
    return image


def generate_centers(rng: RandomState, w: int, h: int, light_dir: List[int], size: Tuple[int, int]) -> List[Point]:
    """Generate shifted centers for the shaded components of a single etch pit."""
    center_x, center_y = rng.randint(0, w), rng.randint(0, h)

    light_vec_3d = normalise_light_direction(light_dir)
    light_vec_2d = light_vec_3d[:2]

    norm_xy = np.linalg.norm(light_vec_2d)
    if norm_xy < 1e-8:
        light_vec_2d = np.array([1.0, 0.0], dtype=np.float32)
    else:
        light_vec_2d /= norm_xy

    step = 0.25 * min(size)
    step_vec = step * light_vec_2d

    # base, shadow toward light, inner at base, bright opposite light
    offsets = [0.0, 1.1, 0.0, -0.75]

    centers = []
    for k in offsets:
        offset = k * step_vec
        x = int(round(center_x + offset[0]))
        y = int(round(center_y + offset[1]))
        centers.append((x, y))

    return centers


def _generate_pit_size(rng: RandomState, etch_pits_size_range: List[int]) -> Tuple[int, int]:
    size_x = rng.randint(*etch_pits_size_range)
    size_y = max(3, size_x - rng.randint(size_x // 3, size_x // 2))
    return size_x, size_y


def add_colored_shapes(image: Image, rng: RandomState, centers: List[Point], size: Tuple[int, int],
                       angle: int, colors: List[int], light_dir: List[int]) -> Image:
    """Draw and blend the shading shape layers forming one etch pit."""
    temp_image = np.zeros_like(image, dtype=np.uint8)

    scales = compute_shapes_scales(light_dir)
    base_center = centers[0]
    base_contour = create_pit_shape(image, base_center, angle, size, scales[0], rng)

    for center, color, scale in zip(centers, colors, scales):
        scaled_contour = _scale_contour(base_contour, scale, base_center)
        dx = center[0] - base_center[0]
        dy = center[1] - base_center[1]
        moved_contour = _translate_contour(scaled_contour, (dx, dy))

        color_w_deviation = max(0, min(255, int(color + rng.randint(-5, 5))))
        fin_color = (color_w_deviation, color_w_deviation, color_w_deviation)
        cv2.drawContours(temp_image, [moved_contour], -1, fin_color, -1)

    blur_size = max(3, int(min(size) * 0.4) | 1)
    blurred = cv2.GaussianBlur(temp_image, (blur_size, blur_size), sigmaX=0)

    mask = (temp_image > 0).astype(np.uint8)
    result = merge_pit_into_image(image, blurred, mask, size)

    return result


def compute_shapes_scales(light_dir: List[int]) -> List[float]:
    light_vec = normalise_light_direction(light_dir)
    lz = abs(float(light_vec[2]))

    z_gain = 0.20
    growth = 1.0 + z_gain * lz

    return [1.00 * growth, 0.85 * growth, 0.76 * growth, 0.70 * growth]


def create_pit_shape(image: Image, center: Point, angle: int, size: Tuple[int, int], scale: float, rng: RandomState)\
        -> Contour:
    """Create an irregular contour for a single etch pit shape."""
    temp_shape_mask = np.zeros_like(image, dtype=np.uint8)
    ellipse_axes = (int(size[0] * scale), int(size[1] * scale))
    color = (255, 255, 255)

    cv2.ellipse(temp_shape_mask, center, ellipse_axes, angle, 0, 360, color, -1)
    contours, _ = cv2.findContours(temp_shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    base_contour = contours[0]

    avg_size = (size[0] + size[1]) // 2
    max_noise = avg_size * 0.5
    smoothing = max_noise * 0.15
    noise = generate_smooth_noise(base_contour, center, rng, max_noise, smoothing)
    base_contour = (base_contour.astype(np.int32) + noise).astype(np.int32)

    return base_contour.reshape(-1, 1, 2)


def generate_smooth_noise(contour: Contour, center: Point, rng: RandomState, max_noise: float, smoothing: float) \
        -> np.ndarray:
    """Generate smooth radial contour perturbations for a pit boundary."""
    pts = contour[:, 0, :].astype(np.float32)

    # unit radial direction at each point
    center = np.array(center, np.float32)
    vec_to_center = pts - center
    radii = np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6
    radial_dir = vec_to_center / radii

    n = len(contour)
    blend_len = max(2, int(n * 0.2))
    core_noise = rng.normal(0.0, max_noise * 2, size=n - blend_len).astype(np.float32)
    t = np.linspace(0, 1, blend_len + 2, dtype=np.float32)[1:-1]
    transition = (1 - t) * core_noise[-1] + t * core_noise[0]
    noise = np.concatenate([core_noise, transition]).astype(np.float32)

    noise_smooth = cv2.GaussianBlur(noise.reshape(-1, 1), (0, 0), sigmaX=smoothing).reshape(-1)
    radial_displacement = radial_dir * noise_smooth.reshape(n, 1)

    return radial_displacement.astype(np.int32)


def _scale_contour(contour: Contour, scale_factor: float, center: Point) -> Contour:
    center_arr = np.array([[center]], dtype=np.float32)
    centered = contour.astype(np.float32) - center_arr
    scaled = centered * scale_factor
    return (scaled + center_arr).astype(np.int32)


def _translate_contour(contour: Contour, delta: Tuple[int, int]) -> Contour:
    dx, dy = delta
    translated = contour + np.array([[[dx, dy]]], dtype=np.int32)
    return translated.astype(np.int32)


def merge_pit_into_image(image: Image, pit: Image, mask: np.ndarray, size: Tuple[int, int]) -> Image:
    """Blend a generated pit into the image using a softened alpha mask."""
    avg_size = (size[0] + size[1]) / 2
    er_size = max(3, int(avg_size * 0.4) | 1)
    temp_mask = cv2.erode(mask, np.ones((er_size, er_size), dtype=np.uint8), iterations=1)

    blur_size = max(3, int(avg_size * 0.3) | 1)
    alpha = cv2.GaussianBlur(temp_mask.astype(np.float32), (blur_size, blur_size), sigmaX=0)

    alpha_max = alpha.max()
    if alpha_max > 0:
        alpha /= alpha_max

    alpha = np.clip(alpha, 0.0, 1.0)

    img_f = image.astype(np.float32)
    pit_f = pit.astype(np.float32)
    hf_strength = 0.5

    low = cv2.GaussianBlur(img_f, (0, 0), 2)
    hf = img_f - low

    # Bleed high-frequency image texture into the pit interior.
    pit_f = np.clip(pit_f + (hf_strength * alpha) * hf, 0.0, 255.0)

    merged = img_f * (1.0 - alpha) + pit_f * alpha
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    return merged
