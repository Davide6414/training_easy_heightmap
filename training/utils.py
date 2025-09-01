import json, os, math, numpy as np
from PIL import Image

def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    img_exts = {".png",".jpg",".jpeg",".tif",".tiff","bmp"}
    if ext in img_exts:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        if arr.max() > 1.0: arr = arr / 255.0
        return arr
    raise ValueError(f"Unsupported file type: {path}")

def save_array(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, arr)
    elif ext in {".png",".jpg",".jpeg",".bmp"}:
        from PIL import Image
        a = arr.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        Image.fromarray((a*255).astype(np.uint8)).save(path)
    elif ext in {".tif",".tiff"}:
        from PIL import Image
        Image.fromarray(arr.astype(np.float32)).save(path, compression=None)
    else:
        raise ValueError(f"Unsupported save type: {path}")

def robust_norm_fit(values):
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75 ,25])
    iqr = float(max(q75 - q25, 1e-6))
    return {"median":med, "iqr":iqr}

def robust_norm_apply(x, stats):
    return (x - stats["median"]) / (stats["iqr"])

def robust_denorm(x, stats):
    return x * stats["iqr"] + stats["median"]

def fit_affine_ab(x, y):
    # Solve y ~= a*x + b in least squares
    X = x.reshape(-1).astype(np.float64)
    Y = y.reshape(-1).astype(np.float64)
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(b)

def hann2d(h, w):
    wx = 0.5*(1-np.cos(2*np.pi*np.arange(w)/(w-1)))
    wy = 0.5*(1-np.cos(2*np.pi*np.arange(h)/(h-1)))
    return np.outer(wy, wx).astype(np.float32)
