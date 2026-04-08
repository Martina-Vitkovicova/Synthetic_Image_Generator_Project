import hashlib
from typing import List, Tuple, Dict
import cv2
import numpy as np
from scipy import ndimage
import skimage
import skimage.draw as skdraw

from perlin_numpy import generate_perlin_noise_2d

from src.finite_voronoi_gen import get_finite_voronoi
from utils.type_definitions import (Image, IndexedRegion, Vertices, Edge, Point, RandomState, Coords, BorderMask,
                                    GradientImage)


def generate_matrix(seed: int, *, img_size: List[int], light_direction: List[int], voronoi_regions_num_range: List[int],
                    voronoi_color_range: List[int], voronoi_border_width: int, voronoi_border_jaggedness: float,
                    voronoi_border_noise: float, voronoi_border_3d: bool, voronoi_shading_color_range: List[int],
                    voronoi_perlin_noise: List[int], **kwargs) -> Image:
    """
    Generate a synthetic grayscale image representing grain-like metal matrix background using Voronoi tessellation.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    img_size : list of int
        Dimensions of the image as [width, height].
    voronoi_regions_num_range : list of int
        Range for the number of Voronoi regions in the image.
    voronoi_color_range : list of int
        Range for the possible intensity of Voronoi regions.
    voronoi_border_width : int
        Width of the Voronoi borders in pixels.
    voronoi_border_jaggedness : float
        Factor controlling the jaggedness of borders between grains, usually in range [0, 1].
    voronoi_border_noise : float
        Noise level for border distortion, usually in range [0, 1].
    voronoi_border_3d : bool
        Whether to apply 3D shading to borders.
    voronoi_light_direction : list of int
        Direction of the light source for 3D shading as triplet [x, y, z].
    voronoi_shading_color_range : list of int
        Grayscale range used for the border shading effect.
    voronoi_perlin_noise : list of int
        Range controlling the amount of Perlin noise added within regions.

    Returns
    -------
    Image
        The generated polycrystalline metal matrix image.
    """
    rng = np.random.RandomState(seed)
    width, height = img_size
    image = np.full((height, width), 140, dtype=np.float32)

    regions, vertices = get_finite_voronoi(voronoi_regions_num_range, rng, img_size)

    modified_borders = distort_borders(regions, vertices, width, height, voronoi_border_jaggedness,
                                       voronoi_border_noise)

    for region in regions:
        y_coords, x_coords = get_distorted_region_coords(modified_borders, region)
        row, col = skdraw.polygon(y_coords, x_coords, shape=image.shape)
        color = rng.randint(*voronoi_color_range)
        image[row, col] = color

        image = improve_borders(image, y_coords, x_coords, voronoi_border_width, rng, color, voronoi_border_3d,
                                light_direction, voronoi_shading_color_range)
        image = add_perlin_noise(image, rng, row, col, voronoi_perlin_noise)

    return image


def distort_borders(regions: List[IndexedRegion], vertices: Vertices, width: int, height: int,
                    voronoi_border_jaggedness: float, voronoi_border_noise: float) -> Dict[Edge, Vertices]:
    """ Transforms borders consisting of straight lines to jagged polylines. """

    modified_borders = {}

    for region in regions:
        for i in range(len(region)):
            edge = (region[i], region[(i + 1) % len(region)])
            sorted_edge = tuple(sorted(edge))

            if sorted_edge not in modified_borders:
                edge_seed = _get_edge_seed(sorted_edge)
                edge_rng = np.random.RandomState(edge_seed)
                v1, v2 = vertices[sorted_edge[0]], vertices[sorted_edge[1]]
                v1, v2 = tuple(map(int, v1)), tuple(map(int, v2))
                modified_borders[sorted_edge] = divide_edge(v1, v2, edge_rng, width, height, voronoi_border_jaggedness,
                                                            voronoi_border_noise)

    return modified_borders


def _get_edge_seed(edge: Edge) -> int:
    """ Create a unique seed to set a random generator with. """

    edge_bytes = str(edge).encode('utf-8')
    return int(hashlib.sha256(edge_bytes).hexdigest(), 16) % (2 ** 32)


def divide_edge(start: Point, end: Point, rng: RandomState, width: int, height: int, voronoi_border_jaggedness: float,
                voronoi_border_noise: float) -> Vertices:
    """ Divides a straight line given by start and end point to number of segments and adds noise to the coordinates
    to make the new line jagged. """

    # if the line doesn't lie inside the image area, it's not necessary to divide it
    # if it lies partially, we divide only the part of the line which lies in the area
    is_inside, new_start, new_end = cv2.clipLine((0, 0, width, height), start, end)
    if not is_inside:
        return np.array([np.array(start), np.array(end)])

    noisy_points = [np.array(new_start)]

    line_length = np.linalg.norm(np.array(new_end) - np.array(new_start))
    base_segments = max(2, int(np.sqrt(line_length) * voronoi_border_jaggedness))
    # changes in x, y coords to make the border jagged
    noise_level = max(3, min(15, int(np.sqrt(line_length) * voronoi_border_noise) + rng.randint(-2, 2)))
    # to how many segments will the line be divided
    num_segments = max(3, min(15, base_segments + rng.randint(-base_segments // 2, base_segments // 2)))
    # no need to add randomness to the spacing because noise creates the randomness
    points = np.linspace(new_start, new_end, num_segments + 1)

    for i in range(1, len(points) - 1):
        noise = rng.normal(0, noise_level, size=2)
        noisy_points.append(points[i] + noise)
    noisy_points.append(np.array(new_end))

    return np.array(noisy_points)


def get_distorted_region_coords(modified_borders: Dict[Edge, Vertices], region: IndexedRegion) -> Tuple[Coords, Coords]:
    """ Return coordinates of the distorted region border. """

    y_coords, x_coords = [], []
    for j in range(len(region)):
        edge = (region[j], region[(j + 1) % len(region)])
        sorted_edge = tuple(sorted(edge))
        poly_edge = modified_borders[sorted_edge]

        if edge != sorted_edge:
            poly_edge = poly_edge[::-1]

        x_coords.extend(poly_edge[:, 0])
        y_coords.extend(poly_edge[:, 1])

    return np.array(y_coords).astype(int), np.array(x_coords).astype(int)


def improve_borders(image: Image, y_coords: Coords, x_coords: Coords, width: int, rng: RandomState, color: int,
                    voronoi_border_3d: bool, voronoi_light_direction: List[int],
                    voronoi_shading_color_range: List[int]) -> Image:
    """ Changes the border appearance - width and color. """

    border_mask = _create_border_mask(image, y_coords, x_coords, width)

    if not voronoi_border_3d:
        new_image = image.copy()
        new_image[border_mask == 1] = color + rng.randint(-15, 15)
    else:
        grad_shading = compute_3d_border_effect(border_mask, rng, voronoi_light_direction, voronoi_shading_color_range)
        border_mask = skimage.morphology.dilation(border_mask, skimage.morphology.disk(width))
        new_image = _blend_borders_w_bg(border_mask, image, grad_shading)

    return new_image


def compute_3d_border_effect(border_mask: BorderMask, rng: RandomState, voronoi_light_direction: List[int],
                             voronoi_shading_color_range: List[int]) -> GradientImage:
    """ Compute borders colors to make it look like 3D edges. """

    height_map = get_height_map(border_mask, 2)
    dx, dy = np.gradient(height_map)
    # dz points perpendicularly away from the surface
    dz = np.ones_like(height_map)
    normals = np.dstack((-dx, -dy, dz))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8  # avoid division by zero

    light_dir = np.array(voronoi_light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # transform the color to specified values range
    gradient_shading = np.dot(normals, light_dir).astype(np.float32)
    gradient_shading = np.clip(gradient_shading, 0, 1)
    gradient_shading = ((gradient_shading - gradient_shading.min()) /
                        (gradient_shading.max() - gradient_shading.min() + 1e-8))

    lower, upper = [x + rng.randint(-10, 10) for x in voronoi_shading_color_range]
    gradient_shading = gradient_shading * (upper - lower) + lower
    gradient_shading = ndimage.gaussian_filter(gradient_shading, sigma=1)

    return gradient_shading


def _create_border_mask(image: Image, y_coords: Coords, x_coords: Coords, width: int) -> BorderMask:
    """ Creates a binary mask of the borders with given width. """
    border_mask = np.zeros_like(image)
    perim_row, perim_col = skdraw.polygon_perimeter(y_coords, x_coords, shape=image.shape)
    border_mask[perim_row, perim_col] = 1

    disk = skimage.morphology.disk(width)
    border_mask = skimage.morphology.dilation(border_mask, disk)

    border_mask = ndimage.gaussian_filter(border_mask, sigma=3)
    border_mask = border_mask > 0.4

    return border_mask.astype(np.int8)


def _blend_borders_w_bg(border_mask: BorderMask, image: Image, grad_shading: GradientImage) -> Image:
    """ Blends the 3d border effect image with the original image according to the height map. """
    height_map = get_height_map(border_mask, 2)
    border_mask = border_mask.astype(bool)
    new_image = image.copy()

    new_image[border_mask] = (height_map[border_mask] * grad_shading[border_mask] +
                              (1 - height_map[border_mask]) * image[border_mask])

    return new_image


def add_perlin_noise(image: Image, rng: RandomState, row: Coords, col: Coords, voronoi_perlin_noise: List[int]) \
        -> Image:
    """ Adds perlin noise the whole background. """

    new_image = image.copy()
    alpha = rng.randint(*voronoi_perlin_noise) / 100
    divisor = rng.choice((16, 32, 64))
    res = (image.shape[0] // divisor, image.shape[1] // divisor)

    height, width = image.shape
    noise = generate_perlin_noise_2d((width, height), res, rng=rng)
    # * 127.5 so the noise mean is 0
    noise_amplitude = alpha * 127.5
    new_image[row, col] = image[row, col] + noise[row, col] * noise_amplitude

    return new_image


def get_height_map(border_mask: BorderMask, sigma: float) -> Image:
    """ Computes an image with high values in the center of voronoi borders which gradually gets smaller allowing to use
    it as an alpha mask for blending the voronoi borders to the image. """

    distance_map = ndimage.distance_transform_edt(border_mask)
    height_map = ndimage.gaussian_filter(distance_map, sigma=sigma)

    # normalize the height map to range [0, 1]
    max_distance = np.max(height_map)
    height_map = height_map / max_distance if max_distance > 0 else height_map

    return height_map
