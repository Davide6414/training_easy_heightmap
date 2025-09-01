import os, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from utils import load_array, robust_norm_apply

class PairDataset(Dataset):
    def __init__(self, data_root, split_file, stats, a, b, context=320, crop=256, augment=True):
        self.data_root = data_root
        self.c_dir = os.path.join(data_root, "c")
        self.h_dir = os.path.join(data_root, "hr")
        with open(split_file, "r", encoding="utf-8") as f:
            self.names = [ln.strip() for ln in f if ln.strip()]
        self.stats = stats
        self.a, self.b = a, b
        self.context, self.crop = context, crop
        self.augment = augment

    def __len__(self): return len(self.names)

    def _load_pair(self, name):
        low = load_array(os.path.join(self.c_dir, name))
        high = load_array(os.path.join(self.h_dir, name))
        # ensure float32
        low = low.astype(np.float32); high = high.astype(np.float32)
        return low, high

    def _random_crop(self, arr, size):
        H, W = arr.shape
        if H < size or W < size:
            # pad reflectively
            ph = max(0, size - H); pw = max(0, size - W)
            arr = np.pad(arr, ((0,ph),(0,pw)), mode='reflect')
            H, W = arr.shape
        y = 0 if H == size else np.random.randint(0, H - size + 1)
        x = 0 if W == size else np.random.randint(0, W - size + 1)
        return arr[y:y+size, x:x+size]

    def __getitem__(self, idx):
        name = self.names[idx]
        low, high = self._load_pair(name)

        # context crop then center-crop to 256
        ctx = self._random_crop(low, self.context)
        # ensure same crop on high
        # For simplicity: recrop both from same random origin
        # Repeat selection:
        H, W = low.shape
        if H < self.context or W < self.context:
            ph = max(0, self.context - H); pw = max(0, self.context - W)
            low_p  = np.pad(low,  ((0,ph),(0,pw)),  mode='reflect')
            high_p = np.pad(high, ((0,ph),(0,pw)), mode='reflect')
        else:
            low_p, high_p = low, high
        y = np.random.randint(0, low_p.shape[0]-self.context+1)
        x = np.random.randint(0, low_p.shape[1]-self.context+1)
        low_ctx  = low_p[y:y+self.context,  x:x+self.context]
        high_ctx = high_p[y:y+self.context, x:x+self.context]

        # center-crop 256
        s, t = self.context, self.crop
        ys, xs = (s - t)//2, (s - t)//2
        low256  = low_ctx[ys:ys+t, xs:xs+t]
        high256 = high_ctx[ys:ys+t, xs:xs+t]

        # simple flips augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                low256 = np.flip(low256, axis=1); high256 = np.flip(high256, axis=1)
            if np.random.rand() < 0.5:
                low256 = np.flip(low256, axis=0); high256 = np.flip(high256, axis=0)

        low_aligned = self.a * low256 + self.b
        target_res  = high256 - low_aligned

        low_n  = robust_norm_apply(low256, self.stats["low"])
        res_n  = robust_norm_apply(target_res, self.stats["res"])
        high_n = robust_norm_apply(high256, self.stats["high"])

        # tensors [1,H,W]
        to_t = lambda a: torch.from_numpy(a[None, ...].astype('float32'))
        return to_t(low_n), to_t(robust_norm_apply(low_aligned, self.stats["low"])), to_t(high_n), to_t(res_n)
