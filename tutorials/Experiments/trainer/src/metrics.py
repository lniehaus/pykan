
import numpy as np
from skimage import measure

# 1. Counting Connected Regions (approximate by counting contiguous regions in prediction map)
def count_connected_regions(pred_map):
    # pred_map: 2D array of predicted class labels
    # Count contiguous regions for each class, sum over all classes
    regions = 0
    for c in np.unique(pred_map):
        labeled = measure.label(pred_map == c, connectivity=1)
        regions += labeled.max()
    return regions

# mlp_linear_regions = count_linear_regions(mlp_pred)
# kan_linear_regions = count_linear_regions(kan_pred)

# 2. Decision Boundary Length (2D)
def calc_boundary_length(pred_map, xx, yy):
    # Find boundaries between classes
    contours = measure.find_contours(pred_map, 0.5)
    total_length = 0
    for contour in contours:
        # Convert contour indices to x/y coordinates
        x_coords = np.interp(contour[:, 1], np.arange(xx.shape[1]), xx[0])
        y_coords = np.interp(contour[:, 0], np.arange(yy.shape[0]), yy[:, 0])
        points = np.stack([x_coords, y_coords], axis=1)
        seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length += seg_lengths.sum()
    return total_length

# mlp_boundary_length = boundary_length(mlp_pred, xx, yy)
# kan_boundary_length = boundary_length(kan_pred, xx, yy)

# 3. Curvature Measures (approximate: mean absolute angle change along boundary)
def calc_boundary_curvature(pred_map, xx, yy):
    contours = measure.find_contours(pred_map, 0.5)
    total_curvature = 0
    total_points = 0
    for contour in contours:
        x_coords = np.interp(contour[:, 1], np.arange(xx.shape[1]), xx[0])
        y_coords = np.interp(contour[:, 0], np.arange(yy.shape[0]), yy[:, 0])
        points = np.stack([x_coords, y_coords], axis=1)
        if len(points) < 3:
            continue
        v1 = points[1:-1] - points[:-2]
        v2 = points[2:] - points[1:-1]
        # Normalize
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
        dot = np.clip((v1 * v2).sum(axis=1), -1, 1)
        angles = np.arccos(dot)
        total_curvature += np.abs(angles).sum()
        total_points += len(angles)
    mean_curvature = total_curvature / (total_points + 1e-8)
    return mean_curvature

# mlp_curvature = boundary_curvature(mlp_pred, xx, yy)
# kan_curvature = boundary_curvature(kan_pred, xx, yy)

# 4. Fractal Dimension (Box-counting)
def calc_fractal_dimension(pred):
    # pred: 2D binary image of boundary
    def boxcount(pred, k):
        S = np.add.reduceat(
            np.add.reduceat(pred, np.arange(0, pred.shape[0], k), axis=0),
                               np.arange(0, pred.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    # Extract boundary
    boundary = np.zeros_like(pred, dtype=bool)
    for c in np.unique(pred):
        mask = (pred == c)
        eroded = np.zeros_like(mask)
        eroded[1:-1,1:-1] = mask[1:-1,1:-1] & mask[:-2,1:-1] & mask[2:,1:-1] & mask[1:-1,:-2] & mask[1:-1,2:]
        boundary |= mask ^ eroded
    Z = boundary.astype(np.uint8)
    # Minimal dimension of image
    p = min(pred.shape)
    n = 2**np.floor(np.log2(p)).astype(int)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(pred, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# mlp_fractal_dim = fractal_dimension(mlp_pred)
# kan_fractal_dim = fractal_dimension(kan_pred)

