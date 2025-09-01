import argparse, os, json, numpy as np, matplotlib.pyplot as plt
from utils import load_array
from infer import predict_tile
from model import SRUNet
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.load(open(os.path.join(args.data_root,"stats.json"),"r",encoding="utf-8"))
    a = stats["affine"]["a"]; b = stats["affine"]["b"]
    net = SRUNet(base=48).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval()

    low = load_array(os.path.join(args.data_root,"c", args.name))
    high = load_array(os.path.join(args.data_root,"hr", args.name))

    pred = predict_tile(net, low, stats, a, b, device)

    plt.figure(figsize=(12,4))
    for i,(arr,title) in enumerate([(low,"low"),(high,"high"),(pred,"pred")]):
        plt.subplot(1,3,i+1); plt.title(title); plt.imshow(arr); plt.colorbar()
    plt.tight_layout(); plt.show()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", default="runs/best.pt")
    ap.add_argument("--name", required=True, help="filename present in splits")
    args = ap.parse_args()
    main(args)
