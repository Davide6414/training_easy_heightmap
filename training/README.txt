Dem SR — 256x256 low→high, tile-safe
====================================

Folders expected on disk (your machine):
- data/c       : low-detail tiles (e.g., .npy/.tif/.png)
- data/hr      : high-detail tiles
- data/splits  : train.txt / val.txt / test.txt listing filenames WITHOUT folder prefix

Key steps:
1) compute_stats.py  -> fits global a,b (high ≈ a*low + b) and robust stats (median, IQR). Saves stats.json
2) train.py          -> trains SRUNet on residual; logs to runs/
3) infer.py          -> sliding-window inference with Hann blending for large mosaics
4) sanity_plot.py    -> optional quick check on a few pairs

All code assumes grayscale single-channel inputs sized 256x256. If not, it will center-crop or resize.
Run examples (PowerShell from project root):
  python compute_stats.py --data-root "C:/Users/dadic/Desktop/programmi/mappeGenerator/model_elevation/data"
  python train.py --data-root "C:/Users/dadic/Desktop/programmi/mappeGenerator/model_elevation/data" --epochs 100
  python infer.py --data-root "C:/Users/dadic/Desktop/programmi/mappeGenerator/model_elevation/data" --in "C:/path/to/big_low.npy" --out "C:/path/to/out.npy"
