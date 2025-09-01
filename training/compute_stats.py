import argparse, os, numpy as np, json
from utils import load_array, robust_norm_fit, fit_affine_ab

def main(args):
    c_dir = os.path.join(args.data_root, "c")
    h_dir = os.path.join(args.data_root, "hr")
    splits = os.path.join(args.data_root, "splits", "train.txt")
    names = [ln.strip() for ln in open(splits,'r',encoding='utf-8') if ln.strip()]
    vals_low = []
    vals_high = []
    # fit a,b on a random subset to be quick
    sub = names[::max(1, len(names)//2000)]  # ~1000 samples
    a_sum, b_sum, k = 0.0, 0.0, 0
    for nm in sub:
        low = load_array(os.path.join(c_dir, nm)).astype('float32')
        high = load_array(os.path.join(h_dir, nm)).astype('float32')
        a,b = fit_affine_ab(low, high)
        a_sum += a; b_sum += b; k += 1
        vals_low.append(low.flatten()); vals_high.append(high.flatten())
    a_glob = a_sum / max(k,1); b_glob = b_sum / max(k,1)
    vals_low = np.concatenate(vals_low); vals_high = np.concatenate(vals_high)
    # residual for stats
    res = (vals_high - (a_glob*vals_low + b_glob))
    stats = {
        "low":  robust_norm_fit(vals_low),
        "high": robust_norm_fit(vals_high),
        "res":  robust_norm_fit(res),
        "affine": {"a": float(a_glob), "b": float(b_glob)}
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved", os.path.join(args.out_dir, "stats.json"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    main(args)
