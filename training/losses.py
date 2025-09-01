import torch, torch.nn.functional as F

def grad_l1(pred, target):
    gx = torch.tensor([[0,-1,1]], dtype=pred.dtype, device=pred.device).view(1,1,1,3)
    gy = torch.tensor([[0],[-1],[1]], dtype=pred.dtype, device=pred.device).view(1,1,3,1)
    px = F.conv2d(pred, gx, padding=(0,1)); py = F.conv2d(pred, gy, padding=(1,0))
    tx = F.conv2d(target, gx, padding=(0,1)); ty = F.conv2d(target, gy, padding=(1,0))
    return (px-tx).abs().mean() + (py-ty).abs().mean()
