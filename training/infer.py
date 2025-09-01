import argparse, os, json, numpy as np, torch
from utils import load_array, save_array, hann2d, robust_norm_apply, robust_denorm
from model import SRUNet

def predict_tile(net, low256, stats, a, b, device):
    low256 = low256.astype('float32')
    low_aligned = a*low256 + b
    low_n = robust_norm_apply(low256, stats["low"])
    low_t = torch.from_numpy(low_n[None,None,...]).to(device)
    with torch.no_grad():
        resid = net(low_t).cpu().numpy()[0,0]
    pred_n = robust_norm_apply(low_aligned, stats["low"]) + resid
    pred = robust_denorm(pred_n, stats["high"])
    return pred

def mosaic_predict(net, arr, stats, a, b, tile=256, stride=192, device='cpu'):
    H, W = arr.shape
    Wwin = hann2d(tile, tile)
    acc = np.zeros_like(arr, dtype=np.float32)
    wgt = np.zeros_like(arr, dtype=np.float32)
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            patch = arr[y:y+tile, x:x+tile]
            pred = predict_tile(net, patch, stats, a, b, device)
            acc[y:y+tile, x:x+tile] += pred * Wwin
            wgt[y:y+tile, x:x+tile] += Wwin
    # handle borders if needed
    eps = 1e-8
    return acc / (wgt + eps)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.load(open(os.path.join(args.data_root,"stats.json"),"r",encoding="utf-8"))
    a = stats["affine"]["a"]; b = stats["affine"]["b"]
    net = SRUNet(base=args.base).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval()

    arr = load_array(args.in_path).astype('float32')
    out = mosaic_predict(net, arr, stats, a, b, tile=256, stride=args.stride, device=device)
    save_array(args.out_path, out)
    print("Saved:", args.out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", default="runs/best.pt")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--base", type=int, default=48)
    args = ap.parse_args()
    main(args)
