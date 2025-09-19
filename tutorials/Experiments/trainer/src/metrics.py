
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

def count_artifacts(pred_map, train_points):
    """
    Count connected regions in pred_map that do not contain any train_points.
    pred_map: 2D array of predicted class labels.
    train_points: Nx2 array of (row, col) indices of training data in pred_map coordinates.
    """
    artifact_count = 0
    train_mask = np.zeros_like(pred_map, dtype=bool)
    for pt in train_points:
        r, c = int(pt[0]), int(pt[1])
        if 0 <= r < pred_map.shape[0] and 0 <= c < pred_map.shape[1]:
            train_mask[r, c] = True
    for c in np.unique(pred_map):
        labeled = measure.label(pred_map == c, connectivity=1)
        for region_idx in range(1, labeled.max() + 1):
            region_mask = labeled == region_idx
            if not np.any(train_mask & region_mask):
                artifact_count += 1
    return artifact_count

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

# FLOPs_MAP = {
#     "zero": 0,
#     "identity": 0,
#     "relu": 1,
#     'square_relu': 2,
#     "sigmoid":4,
#     "silu":5,
#     "tanh":6,
#     "gelu": 14,
#     "polynomial2": 1+2+3-1,
#     "polynomial3": 1+2+3+4-1,
#     "polynomial5": 1+2+3+4+5-1,
# }

# def layer_flops(self, din, dout, shortcut_name, grid, k):
#     flops = (din * dout) * (9 * k * (grid +  1.5 * k) + 2 * grid - 2.5 * k + 1)
#     if shortcut_name == "zero":
#         shortcut_flops = 0
#     else:
#         shortcut_flops = FLOPs_MAP[shortcut_name] * din + 2 * din * dout
#     return flops + shortcut_flops

# def layer_parameters(self, din, dout, shortcut_name, grid, k):
#     parameters = din * dout * (grid + k + 2) + dout
#     if shortcut_name == "zero":
#         shortcut_parameters = 0
#     else:
#         shortcut_parameters = din * dout
#     return parameters + shortcut_parameters

# def total_flops(self):
#     total_flops = 0
#     for i in range(len(self.layers_width) - 1):
#         total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
#     return total_flops

# def total_parameters(self):
#     total_parameters = 0
#     for i in range(len(self.layers_width) - 1):
#         total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
#     return total_parameters

# FLOPs_MAP = {
#     "zero": 0,
#     "identity": 0,
#     "relu": 1,
#     'square_relu': 2,
#     "sigmoid": 4,
#     "silu": 5,
#     "tanh": 6,
#     "gelu": 14,
#     "polynomial2": 1 + 2 + 3 - 1,
#     "polynomial3": 1 + 2 + 3 + 4 - 1,
#     "polynomial5": 1 + 2 + 3 + 4 + 5 - 1,
# }

# def layer_flops(din, dout, shortcut_name, grid, k):
#     flops = (din * dout) * (9 * k * (grid + 1.5 * k) + 2 * grid - 2.5 * k + 1)
#     if shortcut_name == "zero":
#         shortcut_flops = 0
#     else:
#         shortcut_flops = FLOPs_MAP.get(shortcut_name, 0) * din + 2 * din * dout
#     return flops + shortcut_flops

# def layer_parameters(din, dout, shortcut_name, grid, k):
#     print("din:", din, "dout:", dout, "grid:", grid, "k:", k)
#     parameters = din * dout * (grid + k + 2) + dout
#     if shortcut_name == "zero":
#         shortcut_parameters = 0
#     else:
#         shortcut_parameters = din * dout
#     return parameters + shortcut_parameters

# def total_flops(model):
#     total_flops = 0
#     # Try to get required attributes from model
#     layers_width = getattr(model, "layers_width", None)
#     shortcut_function_name = getattr(model, "shortcut_function_name", "zero")
#     grid = getattr(model, "grid", 1)
#     k = getattr(model, "k", 1)
#     if layers_width is None and hasattr(model, "width"):
#         layers_width = model.width
#     if layers_width is None:
#         raise ValueError("Model must have 'layers_width' or 'width' attribute.")
#     for i in range(len(layers_width) - 1):
#         total_flops += layer_flops(layers_width[i][0], layers_width[i+1][0], shortcut_function_name, grid, k)
#     return total_flops

# def total_parameters(model):
#     total_parameters = 0
#     layers_width = getattr(model, "layers_width", None)
#     shortcut_function_name = getattr(model, "shortcut_function_name", "zero")
#     grid = getattr(model, "grid", 1)
#     k = getattr(model, "k", 1)
#     if layers_width is None and hasattr(model, "width"):
#         layers_width = model.width
#     if layers_width is None:
#         raise ValueError("Model must have 'layers_width' or 'width' attribute.")
#     for i in range(len(layers_width) - 1):
#         total_parameters += layer_parameters(layers_width[i][0], layers_width[i+1][0], shortcut_function_name, grid, k)
#     return total_parameters

FLOPs_MAP = {
    "zero": 0,
    "identity": 0,
    "relu": 1,
    'square_relu': 2,
    "sigmoid": 4,
    "silu": 5,
    "tanh": 6,
    "gelu": 14,
    "polynomial2": 1 + 2 + 3 - 1,
    "polynomial3": 1 + 2 + 3 + 4 - 1,
    "polynomial5": 1 + 2 + 3 + 4 + 5 - 1,
}

def layer_flops(din, dout, shortcut_name, grid, k):
    flops = (din * dout) * (9 * k * (grid + 1.5 * k) + 2 * grid - 2.5 * k + 1)
    if shortcut_name == "zero":
        shortcut_flops = 0
    else:
        shortcut_flops = FLOPs_MAP.get(shortcut_name, 0) * din + 2 * din * dout
    return flops + shortcut_flops

def layer_parameters(din, dout, shortcut_name, grid, k):
    print("din:", din, "dout:", dout, "grid:", grid, "k:", k)
    parameters = din * dout * (grid + k + 2) + dout
    if shortcut_name == "zero":
        shortcut_parameters = 0
    else:
        shortcut_parameters = din * dout
    return parameters + shortcut_parameters

def total_flops(model):
    total_flops = 0
    for act_fun in model.act_fun:
        total_flops += layer_flops(act_fun.in_dim, act_fun.out_dim, act_fun.base_fun, act_fun.num, act_fun.k)
    return total_flops

def total_parameters(model):
    total_parameters = 0
    for act_fun in model.act_fun:
        total_parameters += layer_parameters(act_fun.in_dim, act_fun.out_dim, act_fun.base_fun, act_fun.num, act_fun.k)
    return total_parameters



