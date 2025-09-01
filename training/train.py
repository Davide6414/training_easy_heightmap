import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PairDataset
from model import SRUNet
from losses import grad_l1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.load(open(os.path.join(args.data_root,"stats.json"),"r",encoding="utf-8"))
    a = stats["affine"]["a"]; b = stats["affine"]["b"]

    ds_train = PairDataset(args.data_root, os.path.join(args.data_root,"splits","train.txt"), stats, a, b, augment=True)
    ds_val   = PairDataset(args.data_root, os.path.join(args.data_root,"splits","val.txt"),   stats, a, b, augment=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # istanzia il modello senza depth
    net = SRUNet(base=args.base).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = 1e9
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        net.train(); total = 0.0
        for low, low_aligned_n, high_n, res_n in dl_train:
            low, low_aligned_n, high_n, res_n = [t.to(device, non_blocking=True) for t in (low, low_aligned_n, high_n, res_n)]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                resid_pred = net(low)
                pred = low_aligned_n + resid_pred
                l_rec = F.l1_loss(pred, high_n)
                l_res = F.l1_loss(resid_pred, res_n)
                l_grad = grad_l1(pred, high_n) * 0.5
                loss = l_rec + 0.5*l_res + l_grad
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += loss.item() * low.size(0)
        train_loss = total / len(ds_train)

        # validation
        net.eval(); tot = 0.0
        with torch.no_grad():
            for low, low_aligned_n, high_n, res_n in dl_val:
                low, low_aligned_n, high_n = [t.to(device) for t in (low, low_aligned_n, high_n)]
                resid_pred = net(low)
                pred = low_aligned_n + resid_pred
                tot += F.l1_loss(pred, high_n, reduction='sum').item()
        val_mae = tot / len(ds_val) / 256 / 256

        if val_mae < best:
            best = val_mae
            torch.save(net.state_dict(), os.path.join(args.out_dir, "best.pt"))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_MAE_px={val_mae:.6f}  best={best:.6f}")

    torch.save(net.state_dict(), os.path.join(args.out_dir, "last.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=48)
    args = ap.parse_args()
    main(args)
