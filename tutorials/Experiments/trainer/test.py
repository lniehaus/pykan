import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from skimage import measure

def ensure_float64_2d_contig(A):
    """Ensure array is float64, 2D, contiguous, not object."""
    A = np.asarray(A)
    if A.dtype == np.dtype('O'):
        A = A.astype(np.float64)
    elif np.issubdtype(A.dtype, np.bool_) or np.issubdtype(A.dtype, np.integer):
        A = A.astype(np.float64)
    if A.ndim > 2:
        A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError(f"Array must be 2D for plotting/contours. Got shape {A.shape}")
    A = np.ascontiguousarray(A)
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    return A

np.random.seed(0)
torch.manual_seed(0)

# DATA
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ReLUMLP(nn.Module):
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

class Spline1D(nn.Module):
    def __init__(self, n_knots=10, xmin=-2.0, xmax=3.0):
        super().__init__()
        self.n_knots = n_knots
        self.xmin = xmin
        self.xmax = xmax
        self.knots = nn.Parameter(torch.linspace(xmin, xmax, n_knots), requires_grad=False)
        self.values = nn.Parameter(torch.rand(n_knots))
    def forward(self, x):
        x = torch.clamp(x, self.xmin, self.xmax)
        idx_f = (x - self.xmin) / (self.xmax - self.xmin) * (self.n_knots - 1)
        idx0 = torch.floor(idx_f).long()
        idx1 = torch.clamp(idx0 + 1, max=self.n_knots - 1)
        idx0 = torch.clamp(idx0, max=self.n_knots - 2)
        x0 = self.knots[idx0]
        x1 = self.knots[idx1]
        y0 = self.values[idx0]
        y1 = self.values[idx1]
        t = (x - x0) / (x1 - x0 + 1e-8)
        return y0 + t * (y1 - y0)

class SimpleKAN(nn.Module):
    def __init__(self, n_hidden=32, n_knots=16):
        super().__init__()
        self.lin1 = nn.Linear(2, n_hidden)
        self.acts1 = nn.ModuleList([Spline1D(n_knots) for _ in range(n_hidden)])
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.acts2 = nn.ModuleList([Spline1D(n_knots) for _ in range(n_hidden)])
        self.lin3 = nn.Linear(n_hidden, 1)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.stack([act(x[:,i]) for i, act in enumerate(self.acts1)], dim=1)
        x = self.lin2(x)
        x = torch.stack([act(x[:,i]) for i, act in enumerate(self.acts2)], dim=1)
        x = self.lin3(x)
        return x

def train(model, X_train, y_train, epochs=170, lr=1e-2, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losser = nn.BCEWithLogitsLoss()
    X_tensor = torch.tensor(X_train)
    y_tensor = torch.tensor(y_train).float().unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = losser(out, y_tensor)
        loss.backward()
        optimizer.step()
        if verbose and (epoch+1) % 40 == 0:
            print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")

mlp = ReLUMLP(n_hidden=32)
train(mlp, X_train, y_train, epochs=200)
kan = SimpleKAN(n_hidden=32, n_knots=16)
train(kan, X_train, y_train, epochs=200)

def count_linear_regions_relu(model, X, grid_size=120):
    model.eval()
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    coords = np.c_[xx.ravel(), yy.ravel()]
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    with torch.no_grad():
        activations = []
        X_in = coords_tensor
        for layer in model.net:
            X_in = layer(X_in)
            if isinstance(layer, nn.ReLU):
                activations.append((X_in > 0).cpu().numpy())
        if activations:
            pattern_code = np.concatenate([a.astype(np.uint8) for a in activations], axis=1)
            pattern_code = pattern_code.astype(str)
            keys = np.array([''.join(row) for row in pattern_code])
            n_regions = len(np.unique(keys))
        else:
            n_regions = 1
    return n_regions

def estimate_boundary_length(model, X, resolution=400, plot=False, modelname="Model"):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_torch = torch.tensor(grid).float()
    with torch.no_grad():
        logits = model(grid_torch).cpu().numpy()
    logits = logits.reshape(xx.shape)
    preds = (logits > 0)
    preds_plot = ensure_float64_2d_contig(preds)
    contours = measure.find_contours(preds_plot, 0.5)
    total_length = 0.0
    if plot:
        plt.figure(figsize=(7,7))
        plt.title(f"{modelname}: Decision Boundary (length)")
        plt.imshow(preds_plot, cmap='RdBu', interpolation='nearest',
                   extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.3)
    for contour in contours:
        xs = x_min + (x_max - x_min) * contour[:,1]/(resolution-1)
        ys = y_min + (y_max - y_min) * contour[:,0]/(resolution-1)
        points = np.stack([xs, ys], axis=1)
        seg_lens = np.sqrt(np.sum((points[1:] - points[:-1])**2, axis=1))
        total_length += seg_lens.sum()
        if plot:
            plt.plot(xs, ys, 'k')
    if plot:
        Xplot = np.asarray(X, dtype=np.float64)
        plt.scatter(Xplot[:,0], Xplot[:,1], c='gray', s=4, alpha=0.5)
        plt.xlabel('x1'); plt.ylabel('x2')
        filename = f"{modelname.replace(' ','_').lower()}_boundary_length.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    return total_length

def estimate_boundary_curvature(model, X, resolution=400, plot=False, modelname="Model"):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_torch = torch.tensor(grid).float()
    with torch.no_grad():
        logits = model(grid_torch).cpu().numpy()
    logits = logits.reshape(xx.shape)
    preds = (logits > 0)
    preds_plot = ensure_float64_2d_contig(preds)
    contours = measure.find_contours(preds_plot, 0.5)
    total_curvature = 0.0
    if plot:
        plt.figure(figsize=(7,7))
        plt.title(f"{modelname}: Decision Boundary (Curvature)")
        plt.imshow(preds_plot, cmap='RdBu', interpolation='nearest',
                   extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.2)
    for contour in contours:
        xs = x_min + (x_max - x_min) * contour[:,1]/(resolution-1)
        ys = y_min + (y_max - y_min) * contour[:,0]/(resolution-1)
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
        kappa = kappa[np.isfinite(kappa)]
        total_curvature += np.sum(kappa) * ((np.diff(xs)**2 + np.diff(ys)**2)**.5).mean()
        if plot:
            plt.plot(xs, ys, 'k')
    if plot:
        Xplot = np.asarray(X, dtype=np.float64)
        plt.scatter(Xplot[:,0], Xplot[:,1], c='gray', s=4, alpha=0.5)
        plt.xlabel('x1'); plt.ylabel('x2')
        filename = f"{modelname.replace(' ','_').lower()}_boundary_curvature.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    return total_curvature

def boundary_box_counting_dimension(model, X, resolution=400, box_sizes=[2,4,8,16,32,64,128,256]):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_torch = torch.tensor(grid).float()
    with torch.no_grad():
        logits = model(grid_torch).cpu().numpy()
    logits = logits.reshape(xx.shape)
    preds = (logits > 0)
    preds_plot = ensure_float64_2d_contig(preds)
    edge_h = np.abs(np.diff(preds_plot, axis=0))       # (resolution-1, resolution)
    edge_v = np.abs(np.diff(preds_plot, axis=1))       # (resolution, resolution-1)
    boundary_mask = np.zeros((resolution-1, resolution-1), dtype=bool)
    boundary_mask |= edge_h[:, :-1].astype(bool)       # shape (resolution-1, resolution-1)
    boundary_mask |= edge_v[:-1, :].astype(bool)       # shape (resolution-1, resolution-1)

    Ns = []
    for box_size in box_sizes:
        sz = boundary_mask.shape[0] // box_size
        N = 0
        for i in range(box_size):
            for j in range(box_size):
                sub = boundary_mask[i*sz:(i+1)*sz, j*sz:(j+1)*sz]
                if np.any(sub):
                    N += 1
        Ns.append(N)
    logsizes = np.log(1./np.array(box_sizes))
    logsN = np.log(np.array(Ns)+1e-8)
    coeffs = np.polyfit(logsizes, logsN, 1)
    D = coeffs[0]
    return D, (logsizes, logsN)

#----- RUN METRICS, PRINT AND PLOT -----
n_regions_mlp = count_linear_regions_relu(mlp, X)
print(f"[MLP] Estimated number of linear regions hit by grid: {n_regions_mlp}")

bl_mlp = estimate_boundary_length(mlp, X, plot=True, modelname="ReLU MLP")
bl_kan = estimate_boundary_length(kan, X, plot=True, modelname="KAN")
print(f"[MLP] Decision boundary length: {bl_mlp:.2f}")
print(f"[KAN] Decision boundary length: {bl_kan:.2f}")

curve_mlp = estimate_boundary_curvature(mlp, X, plot=True, modelname="ReLU MLP")
curve_kan = estimate_boundary_curvature(kan, X, plot=True, modelname="KAN")
print(f"[MLP] Boundary curvature: {curve_mlp:.2f}")
print(f"[KAN] Boundary curvature: {curve_kan:.2f}")

D_mlp, (ls_mlp, ln_mlp) = boundary_box_counting_dimension(mlp, X)
D_kan, (ls_kan, ln_kan) = boundary_box_counting_dimension(kan, X)
print(f"[MLP] Box-counting fractal dimension: {D_mlp:.2f}")
print(f"[KAN] Box-counting fractal dimension: {D_kan:.2f}")

plt.figure()
plt.plot(ls_mlp, ln_mlp, 'o-', label="MLP")
plt.plot(ls_kan, ln_kan, 'o-', label="KAN")
plt.xlabel("log(1/box size)")
plt.ylabel("log(# boxes with boundary)")
plt.legend(); plt.title("Box-counting for Fractal Dimension")
plt.savefig("box_counting_fractal_dimension.png", dpi=150, bbox_inches='tight')
print("Saved: box_counting_fractal_dimension.png")
plt.close()
